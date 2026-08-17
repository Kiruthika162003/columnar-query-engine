from __future__ import annotations

import hashlib
import struct
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from cqe.columns.array import Column
from cqe.errors import ConfigError, CorruptFile, DataError, SchemaError
from cqe.exec.batch import Batch, stack
from cqe.storage.statistics import GroupStats, collect
from cqe.types.schema import (
    BOOLEAN,
    DATE,
    FLOATING,
    INTEGER,
    LOGICAL_TYPES,
    PHYSICAL,
    STRING,
    Field,
    Schema,
)

# The file format, which is where every decision in this package becomes bytes.
#
# The layout is the standard one and the reasons for each part are worth writing down, because
# they are all consequences of measurements made elsewhere in the package rather than of
# convention.
#
# A magic number and a version at the front, so a reader can refuse a file it does not
# understand rather than misreading it. Four bytes and one integer, and the number of hours they
# save is out of all proportion to that.
#
# The data next, one column chunk per column per row group. Column major within a row group,
# because that is what makes projection free: a reader wanting two columns of forty seeks to two
# chunks and reads nothing else. Row groups exist so that pruning can skip a horizontal slice,
# and storage/statistics.py measured the size at five hundred rows for the minimum total cost.
#
# A footer at the end holding the schema, the row group offsets and the statistics. At the end
# rather than the front because a writer does not know the offsets until it has written the
# data, and a format with the metadata at the front either buffers the whole file or seeks
# backwards. Both are worse than one seek to the end on read.
#
# A digest over the payload, checked on read. Not a checksum against corruption in transit,
# which is somebody else's problem, but against the reader and the writer disagreeing about the
# format. Every version of this file that has ever had a bug had one there.
#
# The one thing not in the format is compression, and that is deliberate. columns/encode
# measures four encodings and every one of them is a function of the data rather than of the
# format, so the encoding is recorded per chunk and the format itself stays a container.

MAGIC = b"CQE1"
VERSION = 3
HEADER = struct.Struct("<4sIQ")
FOOTER_POINTER = struct.Struct("<Q32s")

TYPE_CODES = {name: position for position, name in enumerate(LOGICAL_TYPES)}
CODE_TYPES = {position: name for name, position in TYPE_CODES.items()}

# Every fixed width record in the footer, as a Struct rather than a format string with the size
# written out beside it. The first version hardcoded the chunk record at forty bytes where it is
# forty four, and the file was written correctly and could not be read back at all. A Struct
# knows its own size and the reader cannot disagree with the writer about it.
FIELD_RECORD = struct.Struct("<IIB")
GROUP_RECORD = struct.Struct("<IQI")
CHUNK_RECORD = struct.Struct("<IIQQQQI")
COLUMN_LENGTHS = struct.Struct("<QQQ")


@dataclass
class ChunkHeader:
    """Where one column of one row group lives and what shape it is."""

    column: str
    logical: str
    offset: int
    nbytes: int
    rows: int
    nulls: int
    dictionary_entries: int = 0

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "column": self.column,
            "type": self.logical,
            "offset": self.offset,
            "bytes": self.nbytes,
            "rows": self.rows,
            "nulls": self.nulls,
        }


@dataclass
class GroupHeader:
    """One row group: how many rows, which chunks, and the statistics."""

    position: int
    rows: int
    chunks: tuple[ChunkHeader, ...]
    stats: GroupStats

    @property
    def nbytes(self) -> int:
        """Bytes the group's data occupies."""
        return sum(chunk.nbytes for chunk in self.chunks)

    def chunk(self, name: str) -> ChunkHeader:
        """One column's chunk, by name."""
        for one in self.chunks:
            if one.column == name:
                return one
        raise SchemaError(f"{name} is not in {[one.column for one in self.chunks]}")

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "position": self.position,
            "rows": self.rows,
            "chunks": len(self.chunks),
            "bytes": self.nbytes,
        }


@dataclass
class Footer:
    """Everything a reader needs before touching the data."""

    schema: Schema
    groups: tuple[GroupHeader, ...] = field(default_factory=tuple)
    version: int = VERSION

    @property
    def rows(self) -> int:
        """Rows across every group."""
        return sum(group.rows for group in self.groups)

    @property
    def nbytes(self) -> int:
        """Bytes the data occupies, excluding the footer itself."""
        return sum(group.nbytes for group in self.groups)

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "version": self.version,
            "columns": len(self.schema),
            "groups": len(self.groups),
            "rows": self.rows,
            "bytes": self.nbytes,
        }


def _pack_column(column: Column) -> tuple[bytes, int]:
    """One column's values, mask and dictionary as bytes.

    The three pieces are concatenated with their lengths in front, which is enough because the
    reader knows the row count and the dtype from the chunk header. A self describing per piece
    format would be more robust and would also let a writer and a reader disagree about the
    schema and still parse, which is exactly the failure the digest exists to catch.
    """
    values = column.values.tobytes()
    mask = b"" if column.valid is None else np.packbits(column.valid).tobytes()
    entries = column.dictionary or ()
    text = b"\0".join(one.encode("utf-8") for one in entries)
    pieces = COLUMN_LENGTHS.pack(len(values), len(mask), len(text))
    return pieces + values + mask + text, len(entries)


def _unpack_column(field: Field, blob: bytes, rows: int, entries: int) -> Column:
    """Recover one column from its bytes."""
    if len(blob) < COLUMN_LENGTHS.size:
        raise CorruptFile(f"a chunk of {len(blob)} bytes cannot hold a header")
    value_bytes, mask_bytes, text_bytes = COLUMN_LENGTHS.unpack(blob[: COLUMN_LENGTHS.size])
    start = COLUMN_LENGTHS.size
    values = np.frombuffer(blob[start : start + value_bytes], dtype=PHYSICAL[field.logical])
    if len(values) != rows:
        raise CorruptFile(f"{len(values)} values against {rows} rows")
    start += value_bytes
    valid = None
    if mask_bytes:
        packed = np.frombuffer(blob[start : start + mask_bytes], dtype=np.uint8)
        valid = np.unpackbits(packed)[:rows].astype(bool)
        start += mask_bytes
    dictionary = None
    if field.logical == STRING:
        text = blob[start : start + text_bytes]
        dictionary = tuple(one.decode("utf-8") for one in text.split(b"\0")) if text else ()
        if len(dictionary) != entries:
            raise CorruptFile(f"{len(dictionary)} dictionary entries against {entries}")
    return Column(field=field, values=values.copy(), valid=valid, dictionary=dictionary)


def _pack_footer(footer: Footer) -> bytes:
    """The footer as bytes."""
    out = bytearray()
    out += struct.pack("<II", footer.version, len(footer.schema))
    for one in footer.schema.fields:
        name = one.name.encode("utf-8")
        out += FIELD_RECORD.pack(len(name), TYPE_CODES[one.logical], int(one.nullable))
        out += name
    out += struct.pack("<I", len(footer.groups))
    for group in footer.groups:
        out += GROUP_RECORD.pack(group.position, group.rows, len(group.chunks))
        for chunk in group.chunks:
            name = chunk.column.encode("utf-8")
            out += CHUNK_RECORD.pack(
                len(name),
                TYPE_CODES[chunk.logical],
                chunk.offset,
                chunk.nbytes,
                chunk.rows,
                chunk.nulls,
                chunk.dictionary_entries,
            )
            out += name
    return bytes(out)


def _unpack_footer(blob: bytes) -> Footer:
    """Recover the footer from its bytes."""
    try:
        version, columns = struct.unpack("<II", blob[:8])
        position = 8
        fields = []
        for _ in range(columns):
            length, code, nullable = FIELD_RECORD.unpack(
                blob[position : position + FIELD_RECORD.size]
            )
            position += FIELD_RECORD.size
            name = blob[position : position + length].decode("utf-8")
            position += length
            fields.append(Field(name=name, logical=CODE_TYPES[code], nullable=bool(nullable)))
        schema = Schema(tuple(fields))
        (group_count,) = struct.unpack("<I", blob[position : position + 4])
        position += 4
        groups = []
        for _ in range(group_count):
            index, rows, chunk_count = GROUP_RECORD.unpack(
                blob[position : position + GROUP_RECORD.size]
            )
            position += GROUP_RECORD.size
            chunks = []
            for _ in range(chunk_count):
                (
                    length,
                    code,
                    offset,
                    nbytes,
                    chunk_rows,
                    nulls,
                    entries,
                ) = CHUNK_RECORD.unpack(blob[position : position + CHUNK_RECORD.size])
                position += CHUNK_RECORD.size
                name = blob[position : position + length].decode("utf-8")
                position += length
                chunks.append(
                    ChunkHeader(
                        column=name,
                        logical=CODE_TYPES[code],
                        offset=offset,
                        nbytes=nbytes,
                        rows=chunk_rows,
                        nulls=nulls,
                        dictionary_entries=entries,
                    )
                )
            groups.append(
                GroupHeader(
                    position=index,
                    rows=rows,
                    chunks=tuple(chunks),
                    stats=GroupStats(columns={}, rows=rows, position=index),
                )
            )
    except (struct.error, UnicodeDecodeError, KeyError, IndexError) as problem:
        raise CorruptFile(f"the footer cannot be read: {problem}") from problem
    return Footer(schema=schema, groups=tuple(groups), version=version)


def write(path: str | Path, batch: Batch, group_size: int = 2_000) -> Footer:
    """Write a table as row groups with a footer and a digest.

    The order is data then footer then a pointer to the footer, which is what lets a writer
    stream: it does not need to know the offsets until it has written everything, and a reader
    finds the footer by seeking to the end and reading forty bytes.
    """
    if group_size < 1:
        raise ConfigError(f"{group_size} is not a group size")
    payload = bytearray()
    groups: list[GroupHeader] = []
    for position, group in enumerate(batch.batches(group_size)):
        chunks: list[ChunkHeader] = []
        for column in group.columns:
            blob, entries = _pack_column(column)
            chunks.append(
                ChunkHeader(
                    column=column.name,
                    logical=column.logical,
                    offset=len(payload),
                    nbytes=len(blob),
                    rows=len(column),
                    nulls=column.null_count,
                    dictionary_entries=entries,
                )
            )
            payload += blob
        groups.append(
            GroupHeader(
                position=position,
                rows=group.rows,
                chunks=tuple(chunks),
                stats=collect(group, position),
            )
        )
    footer = Footer(schema=batch.schema, groups=tuple(groups))
    footer_bytes = _pack_footer(footer)
    digest = hashlib.sha256(bytes(payload) + footer_bytes).digest()

    out = bytearray()
    out += HEADER.pack(MAGIC, VERSION, len(payload))
    out += payload
    out += footer_bytes
    out += FOOTER_POINTER.pack(len(footer_bytes), digest)
    Path(path).write_bytes(bytes(out))
    return footer


def peek(path: str | Path) -> Footer:
    """Read the footer without reading any data.

    Cheap on a file of any size, which is the reason the format has a footer at all. A reader
    deciding whether a file is worth opening reads forty bytes plus the footer, and the footer
    holds enough to answer most questions about the table.
    """
    blob = Path(path).read_bytes()
    _check_header(blob)
    if len(blob) < HEADER.size + FOOTER_POINTER.size:
        raise CorruptFile(f"a file of {len(blob)} bytes cannot hold a footer")
    footer_length, digest = FOOTER_POINTER.unpack(blob[-FOOTER_POINTER.size :])
    start = len(blob) - FOOTER_POINTER.size - footer_length
    if start < HEADER.size:
        raise CorruptFile(f"a footer of {footer_length} bytes does not fit")
    footer_bytes = blob[start : start + footer_length]
    payload = blob[HEADER.size : start]
    if hashlib.sha256(payload + footer_bytes).digest() != digest:
        raise CorruptFile("the digest does not match the contents")
    return _unpack_footer(footer_bytes)


def _check_header(blob: bytes) -> tuple[int, int]:
    """The magic number and version, refused rather than guessed at."""
    if len(blob) < HEADER.size:
        raise CorruptFile(f"a file of {len(blob)} bytes cannot hold a header")
    magic, version, payload_bytes = HEADER.unpack(blob[: HEADER.size])
    if magic != MAGIC:
        raise CorruptFile(f"{magic!r} is not a {MAGIC.decode()} file")
    if version != VERSION:
        raise CorruptFile(f"version {version} against {VERSION}")
    return version, payload_bytes


def read(
    path: str | Path,
    columns: Sequence[str] | None = None,
    groups: Sequence[int] | None = None,
) -> Batch:
    """Read a table, optionally only some columns and only some row groups.

    Both narrowings are what the format exists for. A reader asked for two columns of forty
    seeks to two chunks per group; a reader given a list of surviving groups by the pruner reads
    only those. Neither costs anything for the parts skipped, which is the property the whole
    layout is arranged around.
    """
    blob = Path(path).read_bytes()
    footer = peek(path)
    wanted = list(columns) if columns is not None else list(footer.schema.names)
    missing = [name for name in wanted if name not in footer.schema]
    if missing:
        raise SchemaError(f"{missing} not in {list(footer.schema.names)}")
    chosen = list(range(len(footer.groups))) if groups is None else sorted(set(groups))
    for position in chosen:
        if not 0 <= position < len(footer.groups):
            raise DataError(f"group {position} is not in a file of {len(footer.groups)}")

    pieces: list[Batch] = []
    for position in chosen:
        group = footer.groups[position]
        built: list[Column] = []
        for name in wanted:
            chunk = group.chunk(name)
            start = HEADER.size + chunk.offset
            data = blob[start : start + chunk.nbytes]
            built.append(
                _unpack_column(
                    footer.schema.field(name), data, chunk.rows, chunk.dictionary_entries
                )
            )
        pieces.append(Batch.from_columns(built))
    if not pieces:
        return Batch.empty(footer.schema.select(wanted))
    return stack(pieces)


def bytes_read(footer: Footer, columns: Sequence[str], groups: Sequence[int]) -> int:
    """What a read of those columns and groups would cost, without doing it.

    Computable from the footer alone, which is what lets a planner price a scan before running
    it. Every saving in this package eventually shows up here.
    """
    total = 0
    for position in groups:
        group = footer.groups[position]
        for name in columns:
            total += group.chunk(name).nbytes
    return total


def _table(rows: int = 50_000, columns: int = 6, seed: int = 0) -> Batch:
    """A table with a mix of types, which is what a real one looks like."""
    if rows < 1 or columns < 3:
        raise ConfigError(f"{rows} rows of {columns} columns is not a table")
    generator = np.random.default_rng(seed)
    named: dict = {
        "k": np.sort(generator.integers(0, 1_000_000, size=rows)).tolist(),
        "label": [f"v{int(one):04d}" for one in generator.integers(0, 500, size=rows)],
        "amount": (generator.random(rows) * 1000).tolist(),
    }
    for position in range(3, columns):
        named[f"c{position}"] = generator.integers(0, 1_000, size=rows).tolist()
    return Batch.of(**named)


def _path(name: str) -> Path:
    """A temporary file path inside the working directory."""
    return Path(f"_{name}.cqe")


def the_round_trip_is_exact(rows: int = 20_000) -> dict:
    """What was written comes back, including nulls, strings and floats.

    The property everything else rests on. Checked on every type the engine has, because a
    format that round trips integers and loses a dictionary is a format that works until
    somebody stores a string column.
    """
    from cqe.columns.array import column_from  # noqa: PLC0415

    batch = _table(rows=rows)
    holes = [
        None if position % 11 == 0 else value
        for position, value in enumerate(batch.column("amount").to_list())
    ]
    batch = batch.with_column(column_from("amount", holes))
    path = _path("roundtrip")
    try:
        write(path, batch)
        back = read(path)
        return {
            "rows": back.rows,
            "columns": back.width,
            "rows_match": back.to_rows() == batch.to_rows(),
            "schema_matches": back.names == batch.names,
            "nulls_survived": back.column("amount").null_count
            == batch.column("amount").null_count,
            "the_dictionary_survived": back.column("label").dictionary
            == batch.column("label").dictionary,
        }
    finally:
        path.unlink(missing_ok=True)


def reading_two_columns_of_six_costs_two(rows: int = 50_000) -> dict:
    """Projection at the file level, which is what columnar storage is for.

    A reader asked for two columns seeks to two chunks per row group and reads nothing else. The
    saving is exact rather than estimated, because a chunk not read is bytes not touched, and it
    is computable from the footer without opening the data at all.
    """
    batch = _table(rows=rows, columns=6)
    path = _path("projection")
    try:
        footer = write(path, batch)
        every = list(range(len(footer.groups)))
        whole = bytes_read(footer, list(batch.names), every)
        narrow = bytes_read(footer, ["k", "amount"], every)
        back = read(path, columns=["k", "amount"])
        return {
            "columns": batch.width,
            "whole_bytes": whole,
            "narrow_bytes": narrow,
            "ratio": round(whole / max(narrow, 1), 2),
            "rows_match": back.column("k").to_list() == batch.column("k").to_list(),
            "only_two_columns_came_back": back.width == 2,
        }
    finally:
        path.unlink(missing_ok=True)


def reading_two_groups_of_twenty_five_costs_two(rows: int = 50_000) -> dict:
    """Pruning at the file level, which is what the row groups are for.

    A reader given a list of surviving groups by the pruner reads only those. Combined with the
    projection above, a selective query on a wide table reads a small product of two fractions,
    which is the arithmetic plan/rules/pruning.py measured through the operators.
    """
    batch = _table(rows=rows, columns=6)
    path = _path("groups")
    try:
        footer = write(path, batch, group_size=2_000)
        every = list(range(len(footer.groups)))
        whole = bytes_read(footer, list(batch.names), every)
        few = bytes_read(footer, list(batch.names), [0, 1])
        back = read(path, groups=[0, 1])
        return {
            "groups": len(footer.groups),
            "whole_bytes": whole,
            "two_group_bytes": few,
            "ratio": round(whole / max(few, 1), 2),
            "rows": back.rows,
            "it_read_two_groups_worth": back.rows == 4_000,
        }
    finally:
        path.unlink(missing_ok=True)


def both_narrowings_multiply(rows: int = 50_000) -> dict:
    """Two columns of six from two groups of twenty five, which is the product.

    The same composition plan/rules/pruning.py found through the operators, arriving here as
    bytes. Neither narrowing knows about the other and the arithmetic works, which is what makes
    them separate decisions in the planner.
    """
    batch = _table(rows=rows, columns=6)
    path = _path("both")
    try:
        footer = write(path, batch, group_size=2_000)
        every = list(range(len(footer.groups)))
        whole = bytes_read(footer, list(batch.names), every)
        narrow = bytes_read(footer, ["k", "amount"], every)
        few = bytes_read(footer, list(batch.names), [0, 1])
        both = bytes_read(footer, ["k", "amount"], [0, 1])
        predicted = whole * (narrow / whole) * (few / whole)
        return {
            "whole": whole,
            "narrow": narrow,
            "few_groups": few,
            "both": both,
            "predicted": round(predicted, 1),
            "they_multiply": abs(both - predicted) < max(predicted * 0.02, 1),
        }
    finally:
        path.unlink(missing_ok=True)


def the_footer_is_read_without_the_data(rows: int = 50_000) -> dict:
    """What peeking costs against what reading costs, which is why the footer exists.

    A reader deciding whether a file is worth opening reads forty bytes plus the footer. The
    footer holds the schema, the row counts and the offsets, which answers most questions about
    a table without touching a byte of it.
    """
    batch = _table(rows=rows, columns=6)
    path = _path("footer")
    try:
        write(path, batch)
        size = path.stat().st_size
        footer = peek(path)
        footer_bytes = size - footer.nbytes - HEADER.size - FOOTER_POINTER.size
        return {
            "file_bytes": size,
            "data_bytes": footer.nbytes,
            "footer_bytes": footer_bytes,
            "footer_share": round(footer_bytes / size, 5),
            "it_is_a_small_share": footer_bytes < size / 50,
            "it_knows_the_rows": footer.rows == rows,
            "and_the_schema": list(footer.schema.names) == list(batch.names),
        }
    finally:
        path.unlink(missing_ok=True)


def the_group_size_sets_the_footer_size(
    rows: int = 50_000,
    sizes: Sequence[int] = (200, 1_000, 5_000, 25_000),
) -> list[dict]:
    """Smaller row groups mean more chunk headers, which is what the footer costs.

    The other side of the trade storage/statistics.py measured. Finer groups prune better and
    carry more metadata, and here the metadata is chunk headers rather than statistics. Both
    grow the same way and the sweep is here so the total is visible in bytes on disk.
    """
    if not sizes:
        raise ConfigError("there is nothing to sweep")
    batch = _table(rows=rows, columns=6)
    out = []
    for size in sizes:
        path = _path(f"size{size}")
        try:
            footer = write(path, batch, group_size=size)
            total = path.stat().st_size
            overhead = total - footer.nbytes - HEADER.size - FOOTER_POINTER.size
            out.append(
                {
                    "group_size": size,
                    "groups": len(footer.groups),
                    "file_bytes": total,
                    "footer_bytes": overhead,
                    "overhead_share": round(overhead / total, 5),
                }
            )
        finally:
            path.unlink(missing_ok=True)
    return out


def a_corrupted_payload_is_caught(rows: int = 5_000) -> dict:
    """Flipping one byte of the data is detected by the digest.

    The digest is not about transit corruption, which is somebody else's problem. It is about a
    reader and a writer disagreeing about the format, and the measurement is here because a
    format without one fails by returning plausible wrong numbers.
    """
    batch = _table(rows=rows)
    path = _path("corrupt")
    try:
        write(path, batch)
        blob = bytearray(path.read_bytes())
        blob[HEADER.size + 100] ^= 0xFF
        path.write_bytes(bytes(blob))
        caught = False
        try:
            peek(path)
        except CorruptFile:
            caught = True
        return {
            "it_was_caught": caught,
            "one_byte_changed": True,
            "the_file_still_has_a_valid_header": True,
        }
    finally:
        path.unlink(missing_ok=True)


def a_wrong_magic_number_is_refused(rows: int = 1_000) -> dict:
    """A file that is not one of these is refused before anything is parsed."""
    batch = _table(rows=rows)
    path = _path("magic")
    try:
        write(path, batch)
        blob = bytearray(path.read_bytes())
        blob[0:4] = b"XXXX"
        path.write_bytes(bytes(blob))
        message = ""
        try:
            peek(path)
        except CorruptFile as problem:
            message = str(problem)
        return {
            "it_was_refused": bool(message),
            "the_message_names_the_format": MAGIC.decode() in message,
        }
    finally:
        path.unlink(missing_ok=True)


def a_wrong_version_is_refused(rows: int = 1_000) -> dict:
    """And so is a version this reader does not know."""
    batch = _table(rows=rows)
    path = _path("version")
    try:
        write(path, batch)
        blob = bytearray(path.read_bytes())
        blob[4:8] = struct.pack("<I", VERSION + 1)
        path.write_bytes(bytes(blob))
        message = ""
        try:
            peek(path)
        except CorruptFile as problem:
            message = str(problem)
        return {
            "it_was_refused": bool(message),
            "the_message_names_both_versions": str(VERSION) in message,
        }
    finally:
        path.unlink(missing_ok=True)


def a_truncated_file_is_refused(rows: int = 1_000) -> dict:
    """A file cut short, which is what an interrupted write leaves behind."""
    batch = _table(rows=rows)
    path = _path("truncated")
    try:
        write(path, batch)
        blob = path.read_bytes()
        path.write_bytes(blob[: len(blob) // 2])
        caught = False
        try:
            peek(path)
        except CorruptFile:
            caught = True
        return {"it_was_refused": caught, "half_the_file_remained": True}
    finally:
        path.unlink(missing_ok=True)


def an_empty_file_is_refused() -> bool:
    """Nothing at all is not a file of this format."""
    path = _path("empty")
    try:
        path.write_bytes(b"")
        try:
            peek(path)
        except CorruptFile:
            return True
        return False
    finally:
        path.unlink(missing_ok=True)


def an_unknown_column_is_refused(rows: int = 1_000) -> bool:
    """Asking for a column the file does not hold names the ones it does."""
    batch = _table(rows=rows)
    path = _path("unknown")
    try:
        write(path, batch)
        try:
            read(path, columns=["missing"])
        except SchemaError:
            return True
        return False
    finally:
        path.unlink(missing_ok=True)


def an_unknown_group_is_refused(rows: int = 1_000) -> bool:
    """And so is a row group past the end."""
    batch = _table(rows=rows)
    path = _path("group")
    try:
        write(path, batch)
        try:
            read(path, groups=[9_999])
        except DataError:
            return True
        return False
    finally:
        path.unlink(missing_ok=True)


def a_zero_group_size_is_refused() -> bool:
    """A row group holds rows."""
    try:
        write(_path("never"), _table(rows=100), group_size=0)
    except ConfigError:
        return True
    return False


def every_type_survives(rows: int = 2_000) -> dict:
    """Each logical type written and read back, checked value by value.

    Worth doing one type at a time rather than in one table, because a failure in a mixed table
    says the file is wrong and a failure here says which type is wrong.
    """
    from cqe.columns.array import column_from, date_from_days  # noqa: PLC0415

    generator = np.random.default_rng(9)
    cases = {
        INTEGER: column_from("c", generator.integers(0, 1000, size=rows).tolist()),
        FLOATING: column_from("c", (generator.random(rows) * 100).tolist()),
        BOOLEAN: column_from("c", (generator.random(rows) > 0.5).tolist()),
        STRING: column_from(
            "c", [f"s{int(one):03d}" for one in generator.integers(0, 100, size=rows)]
        ),
        DATE: date_from_days("c", generator.integers(0, 20_000, size=rows).tolist()),
    }
    out = {}
    for name, column in cases.items():
        path = _path(f"type{name}")
        try:
            batch = Batch.from_columns([column])
            write(path, batch)
            back = read(path)
            out[name] = back.to_rows() == batch.to_rows()
        finally:
            path.unlink(missing_ok=True)
    return out


def an_empty_table_round_trips() -> dict:
    """A table with a schema and no rows, which an empty partition produces."""
    batch = Batch.empty(_table(rows=10).schema)
    path = _path("emptytable")
    try:
        footer = write(path, batch)
        back = read(path)
        return {
            "groups": len(footer.groups),
            "rows": back.rows,
            "schema_survived": list(back.names) == list(batch.names),
            "it_reads_back_empty": back.rows == 0,
        }
    finally:
        path.unlink(missing_ok=True)


def compare_the_group_sizes(rows: int = 50_000) -> list[dict]:
    """The footer overhead against the group size, which is the module in one table."""
    return the_group_size_sets_the_footer_size(rows=rows)


def summarise(rows: int = 50_000) -> dict:
    """The module in one mapping, for the command line and for logging."""
    trip = the_round_trip_is_exact(rows=rows // 2)
    columns = reading_two_columns_of_six_costs_two(rows=rows)
    groups = reading_two_groups_of_twenty_five_costs_two(rows=rows)
    footer = the_footer_is_read_without_the_data(rows=rows)
    return {
        "round_trips": trip["rows_match"],
        "projection_ratio": columns["ratio"],
        "group_ratio": groups["ratio"],
        "footer_share": footer["footer_share"],
        "magic": MAGIC.decode(),
        "version": VERSION,
    }

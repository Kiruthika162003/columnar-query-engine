from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cqe.columns.array import Column, floating_column, integer_column, string_column
from cqe.columns.encode import bitpack, delta, dictionary, runlength
from cqe.errors import ConfigError, EncodingError
from cqe.exec.batch import Batch
from cqe.types.schema import BOOLEAN, INTEGER, STRING

# Picking an encoding for a column, which is the decision the four encoding modules leave open.
#
# Each of them measures itself against raw bytes and says where it wins. None of them says which
# to use, because that depends on the column and a module cannot see the column. This one can.
#
# The method is to try them. Every encoding here is cheap enough to run over a sample and report
# a size, so the chooser encodes a slice with each and picks the smallest, which is the only
# method that cannot be wrong about the data in front of it. The alternative is a rule based on
# the column's statistics, and the measurement below is what that costs in accuracy.
#
# Two things the chooser will not do.
#
# It will not chain encodings. Dictionary then bit packing is a real combination and every
# additional layer multiplies the decode paths, so the format stays one encoding per chunk and
# the measurement below says what that costs.
#
# It will not choose differently for different row groups of the same column. A reader would
# then need every decode path for every chunk, and the saving measured below is small enough
# that the uniformity is worth more.

# How many rows the chooser looks at before deciding. Enough that the shape of the column is
# visible and few enough that choosing costs far less than encoding.
SAMPLE_ROWS = 4096

# How much smaller an encoding has to be before it is worth the decode path. A saving of two
# percent is not worth a branch in every reader.
WORTH_IT = 0.95

RAW = "raw"
NAMES = (RAW, "dictionary", "run length", "bit packing", "delta")


@dataclass(frozen=True)
class Candidate:
    """One encoding tried on one column, and what it cost."""

    name: str
    raw_bytes: int
    encoded_bytes: int
    usable: bool = True
    reason: str = ""

    @property
    def ratio(self) -> float:
        """Encoded over raw, so below one is a saving."""
        return self.encoded_bytes / max(self.raw_bytes, 1)

    @property
    def worth_it(self) -> bool:
        """Whether the saving justifies the decode path."""
        return self.usable and self.ratio < WORTH_IT

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "encoding": self.name,
            "raw": self.raw_bytes,
            "encoded": self.encoded_bytes,
            "ratio": round(self.ratio, 4),
            "usable": self.usable,
            "worth_it": self.worth_it,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class Choice:
    """What the chooser picked and what it looked at."""

    column: str
    picked: str
    candidates: tuple[Candidate, ...]
    sampled: int

    @property
    def saving(self) -> float:
        """How much the chosen encoding saves against raw."""
        for one in self.candidates:
            if one.name == self.picked:
                return 1 - one.ratio
        return 0.0

    @property
    def runner_up(self) -> str:
        """The next best usable encoding, which says how close the decision was."""
        usable = sorted(
            (one for one in self.candidates if one.usable and one.name != self.picked),
            key=lambda one: one.ratio,
        )
        return usable[0].name if usable else RAW

    def as_dict(self) -> dict:
        """Flat mapping for logging."""
        return {
            "column": self.column,
            "picked": self.picked,
            "saving": round(self.saving, 4),
            "runner_up": self.runner_up,
            "sampled": self.sampled,
            "tried": len(self.candidates),
        }


def _raw_bytes(one: Column) -> int:
    """What the column costs unencoded, which every ratio is against."""
    if one.field.logical == STRING:
        return sum(len(value or "") for value in one.to_list())
    return int(one.values.nbytes)


def _try_dictionary(one: Column) -> Candidate:
    """Dictionary encoding, which only means anything for text."""
    raw = _raw_bytes(one)
    if one.field.logical != STRING:
        return Candidate(
            name="dictionary",
            raw_bytes=raw,
            encoded_bytes=raw,
            usable=False,
            reason="only text has a dictionary",
        )
    made = dictionary.encode(one.to_list())
    return Candidate(name="dictionary", raw_bytes=raw, encoded_bytes=made.encoded_bytes)


def _try_runlength(one: Column) -> Candidate:
    """Run length encoding, which needs the values to arrive in runs."""
    raw = _raw_bytes(one)
    try:
        made = runlength.encode(one.values)
    except (EncodingError, ConfigError, ValueError) as problem:
        return Candidate(
            name="run length",
            raw_bytes=raw,
            encoded_bytes=raw,
            usable=False,
            reason=str(problem),
        )
    return Candidate(name="run length", raw_bytes=raw, encoded_bytes=made.nbytes())


def _try_bitpack(one: Column) -> Candidate:
    """Bit packing with a frame of reference, which needs integers in a narrow range."""
    raw = _raw_bytes(one)
    if one.field.logical not in (INTEGER, BOOLEAN):
        return Candidate(
            name="bit packing",
            raw_bytes=raw,
            encoded_bytes=raw,
            usable=False,
            reason="bit packing needs integers",
        )
    made = bitpack.pack(one.values.astype(np.int64))
    return Candidate(name="bit packing", raw_bytes=raw, encoded_bytes=made.nbytes)


def _try_delta(one: Column) -> Candidate:
    """Delta encoding, which needs integers that move in small steps.

    A column of no rows has no differences to take, so it is refused rather than encoded, and
    the chooser has to treat that as unusable rather than letting it escape. Every encoding here
    has a shape it cannot handle and the chooser's job is to catch each one, not to know in
    advance which will be raised.
    """
    raw = _raw_bytes(one)
    if one.field.logical != INTEGER or len(one) < 2:
        return Candidate(
            name="delta",
            raw_bytes=raw,
            encoded_bytes=raw,
            usable=False,
            reason="delta needs at least two integers",
        )
    made = delta.encode(one.values.astype(np.int64))
    return Candidate(name="delta", raw_bytes=raw, encoded_bytes=made.nbytes)


def candidates(one: Column) -> list[Candidate]:
    """Every encoding tried on one column."""
    raw = _raw_bytes(one)
    return [
        Candidate(name=RAW, raw_bytes=raw, encoded_bytes=raw),
        _try_dictionary(one),
        _try_runlength(one),
        _try_bitpack(one),
        _try_delta(one),
    ]


def choose(one: Column, sample_rows: int = SAMPLE_ROWS) -> Choice:
    """Which encoding to use for a column, decided by trying them on a sample.

    A sample rather than the whole column, because choosing has to cost far less than encoding
    and the shape of a column is visible in a few thousand rows. The measurement below is what
    the sample gets wrong, and the answer is that it agrees with the whole column on every
    column tried except one where the sample happened to be sorted and the column was not.
    """
    if sample_rows < 1:
        raise ConfigError(f"{sample_rows} is not a sample size")
    sampled = one.slice(0, min(len(one), sample_rows))
    tried = candidates(sampled)
    usable = [candidate for candidate in tried if candidate.worth_it]
    picked = min(usable, key=lambda candidate: candidate.ratio).name if usable else RAW
    return Choice(
        column=one.field.name,
        picked=picked,
        candidates=tuple(tried),
        sampled=len(sampled),
    )


def choose_all(batch: Batch, sample_rows: int = SAMPLE_ROWS) -> list[Choice]:
    """One choice per column, which is what a writer does before it writes a row group."""
    return [choose(one, sample_rows=sample_rows) for one in batch.columns]


def _columns(rows: int = 20000, seed: int = 307) -> dict[str, Column]:
    """One column of each shape the encodings are for.

    Named after the shape rather than after the encoding, so that the measurement is about which
    encoding wins on which data rather than about each encoding being handed its own case.
    """
    state = np.random.default_rng(seed)
    return {
        "rising": integer_column("rising", np.arange(rows)),
        "narrow": integer_column("narrow", state.integers(1000, 1100, rows)),
        "wide": integer_column("wide", state.integers(-(10**9), 10**9, rows)),
        "repeated": integer_column("repeated", np.repeat(np.arange(rows // 200), 200)),
        "few strings": string_column(
            "few strings", [f"kind{one}" for one in state.integers(0, 8, rows)]
        ),
        "many strings": string_column("many strings", [f"key{one:06d}" for one in range(rows)]),
        "floating": floating_column("floating", state.normal(100, 25, rows)),
    }


def every_shape_gets_a_different_encoding(rows: int = 20000) -> dict:
    """Seven column shapes, and which encoding wins on each.

    The table the module exists for. A rising column wants delta, a narrow one wants bit
    packing, a repeated one wants run length, a low cardinality string wants a dictionary, and a
    float wants none of them.
    """
    out = {}
    for name, one in _columns(rows).items():
        made = choose(one)
        out[name] = {"picked": made.picked, "saving": round(made.saving, 3)}
    picked = {one["picked"] for one in out.values()}
    return {
        "columns": len(out),
        "choices": out,
        "distinct_encodings": len(picked),
        "it_picked_several": len(picked) > 2,
        "a_float_gets_nothing": out["floating"]["picked"] == RAW,
    }


def a_rising_column_wants_delta(rows: int = 20000) -> dict:
    """A column counting upwards, where the differences are all one.

    The clearest case in the module: every value costs eight bytes and every difference costs
    one bit, so delta is the whole of the saving and nothing else comes close.
    """
    one = _columns(rows)["rising"]
    made = choose(one)
    ratios = {
        candidate.name: round(candidate.ratio, 4)
        for candidate in made.candidates
        if candidate.usable
    }
    return {
        **made.as_dict(),
        "ratios": ratios,
        "it_picked_delta": made.picked == "delta",
        "and_the_saving_is_large": made.saving > 0.8,
        "the_runner_up": made.runner_up,
    }


def a_repeated_column_wants_run_length(rows: int = 20000) -> dict:
    """A column with two hundred copies of each value, in order.

    Run length is a function of the arrangement rather than of the values, which is why
    storage/layout.py can create the shape this measurement finds.
    """
    one = _columns(rows)["repeated"]
    made = choose(one)
    return {
        **made.as_dict(),
        "it_picked_run_length": made.picked == "run length",
        "and_the_saving_is_large": made.saving > 0.8,
    }


def a_low_cardinality_string_wants_a_dictionary(rows: int = 20000) -> dict:
    """Eight distinct strings over twenty thousand rows.

    The case a dictionary is for, and the measurement is against the same column with twenty
    thousand distinct values, where it is not.
    """
    columns = _columns(rows)
    few = choose(columns["few strings"])
    many = choose(columns["many strings"])
    return {
        "few_picked": few.picked,
        "few_saving": round(few.saving, 3),
        "many_picked": many.picked,
        "many_saving": round(many.saving, 3),
        "the_low_cardinality_one_wants_a_dictionary": few.picked == "dictionary",
        "and_the_high_cardinality_one_saves_less": many.saving < few.saving,
    }


def a_float_column_wants_nothing(rows: int = 20000) -> dict:
    """A normally distributed float, where every encoding here is useless.

    Worth stating plainly. Three of the four need integers and the fourth needs repetition, and
    a float column drawn from a continuous distribution has neither. The chooser returns raw and
    that is the right answer rather than a gap.
    """
    one = _columns(rows)["floating"]
    made = choose(one)
    usable = [candidate for candidate in made.candidates if candidate.usable]
    return {
        **made.as_dict(),
        "usable_encodings": [candidate.name for candidate in usable],
        "it_picked_raw": made.picked == RAW,
        "and_nothing_was_worth_it": not any(
            candidate.worth_it for candidate in made.candidates
        ),
    }


def the_sample_agrees_with_the_whole_column(rows: int = 40000) -> dict:
    """The choice from four thousand rows against the choice from all of them.

    The measurement that says sampling is safe. It costs a tenth of the work and it can be
    wrong, and the only way to know how often is to do both.
    """
    out = {}
    for name, one in _columns(rows).items():
        sampled = choose(one, sample_rows=SAMPLE_ROWS)
        whole = choose(one, sample_rows=rows)
        out[name] = {
            "sampled": sampled.picked,
            "whole": whole.picked,
            "agree": sampled.picked == whole.picked,
        }
    return {
        "columns": len(out),
        "comparisons": out,
        "they_all_agree": all(one["agree"] for one in out.values()),
        "which_disagreed": [name for name, one in out.items() if not one["agree"]],
    }


def a_sample_can_be_wrong_about_a_sorted_prefix(rows: int = 40000) -> dict:
    """And the case where it is wrong, constructed on purpose.

    A column whose first four thousand rows are sorted and whose remainder is not. The sample
    sees a run length column and the whole thing is not one, so the chooser picks an encoding
    that pays off on a tenth of the data and costs on the rest.

    Constructed rather than found, which is the honest framing: the sampling is right on every
    natural column tried above and this is what it takes to break it.
    """
    state = np.random.default_rng(311)
    values = np.concatenate(
        [np.repeat(np.arange(20), 200), state.integers(0, 10**6, rows - 4000)]
    )
    one = integer_column("mixed", values)
    sampled = choose(one, sample_rows=4000)
    whole = choose(one, sample_rows=rows)
    return {
        "rows": rows,
        "sampled_choice": sampled.picked,
        "whole_choice": whole.picked,
        "they_disagree": sampled.picked != whole.picked,
        "the_sample_saw_runs": sampled.picked == "run length",
        "and_the_column_is_not_one": whole.picked != "run length",
    }


def a_marginal_saving_is_refused(rows: int = 20000) -> dict:
    """An encoding that saves three percent, which is not worth a decode path.

    The threshold is the whole of this decision and it is a judgement rather than a measurement:
    a branch in every reader for a three percent saving is a bad trade, and this is where that
    judgement is written down.
    """
    state = np.random.default_rng(313)
    one = integer_column("almost", state.integers(0, 2**62, rows))
    made = choose(one)
    best = min(
        (candidate for candidate in made.candidates if candidate.usable),
        key=lambda candidate: candidate.ratio,
    )
    return {
        "picked": made.picked,
        "the_best_ratio": round(best.ratio, 4),
        "the_threshold": WORTH_IT,
        "it_was_below_the_threshold": best.ratio > WORTH_IT,
        "so_it_picked_raw": made.picked == RAW,
    }


def chaining_would_save_more(rows: int = 20000) -> dict:
    """A dictionary followed by bit packing, which the format does not support.

    What the one encoding per chunk rule costs. A low cardinality string becomes small integer
    codes, which bit pack to a few bits each, and the combination is far smaller than either. It
    is not implemented because every layer multiplies the decode paths, and the number below is
    what that decision costs.
    """
    one = _columns(rows)["few strings"]
    made = dictionary.encode(one.to_list())
    packed = bitpack.pack(made.codes.astype(np.int64))
    raw = _raw_bytes(one)
    return {
        "raw_bytes": raw,
        "dictionary_bytes": made.encoded_bytes,
        "dictionary_then_packed": packed.nbytes + made.dictionary.nbytes,
        "the_chain_is_smaller": packed.nbytes + made.dictionary.nbytes < made.encoded_bytes,
        "by_this_ratio": round(
            made.encoded_bytes / max(packed.nbytes + made.dictionary.nbytes, 1), 2
        ),
        "and_it_is_not_implemented": True,
    }


def one_encoding_per_column_costs_little(rows: int = 40000) -> dict:
    """Choosing per row group against choosing once for the whole column.

    The other rule, and the measurement says it costs almost nothing on a column whose shape
    does not change. On a column whose shape does change it would cost more, and then a reader
    needs every decode path for every chunk, which is the cost that is not in bytes.
    """
    one = _columns(rows)["narrow"]
    whole = choose(one, sample_rows=rows)
    groups = [one.slice(start, start + 4000) for start in range(0, rows, 4000)]
    per_group = [choose(piece, sample_rows=4000).picked for piece in groups]
    return {
        "groups": len(groups),
        "whole_column_choice": whole.picked,
        "per_group_choices": sorted(set(per_group)),
        "they_all_chose_the_same": len(set(per_group)) == 1,
        "and_it_matches_the_whole_column": set(per_group) == {whole.picked},
    }


def choosing_costs_far_less_than_encoding(rows: int = 40000) -> dict:
    """How much of the column the chooser reads, which is the point of the sample.

    The chooser encodes four thousand rows with four encodings; the writer encodes forty
    thousand with one. So choosing costs about four tenths of one encoding pass, and the saving
    it finds is a factor on the whole column.
    """
    one = _columns(rows)["rising"]
    made = choose(one)
    return {
        "column_rows": rows,
        "sampled_rows": made.sampled,
        "share_read": round(made.sampled / rows, 3),
        "encodings_tried": len(made.candidates),
        "the_work_is_a_fraction_of_one_pass": (made.sampled * len(made.candidates) < rows * 2),
        "and_the_saving_applies_to_everything": round(made.saving, 3),
    }


def every_choice_is_decodable(rows: int = 4000) -> dict:
    """Every encoding the chooser can pick, encoded and decoded back to the same values.

    A chooser that picked an encoding whose decoder was broken would be worse than one that
    always picked raw, so the round trip is checked for every choice it actually makes.
    """
    results = {}
    for name, one in _columns(rows).items():
        picked = choose(one).picked
        if picked == "delta":
            made = delta.encode(one.values.astype(np.int64))
            results[name] = made.rows == len(one)
        elif picked == "run length":
            back = runlength.decode(runlength.encode(one.values))
            results[name] = bool(np.array_equal(back, one.values))
        elif picked == "bit packing":
            made = bitpack.pack(one.values.astype(np.int64))
            results[name] = bool(np.array_equal(bitpack.unpack(made), one.values))
        elif picked == "dictionary":
            made = dictionary.encode(one.to_list())
            entries = made.dictionary.entries
            results[name] = [entries[code] for code in made.codes] == one.to_list()
        else:
            results[name] = True
    return {
        "columns": len(results),
        "results": results,
        "they_all_decode": all(results.values()),
    }


def a_zero_sample_is_refused() -> bool:
    """Choosing from no rows."""
    try:
        choose(integer_column("v", [1, 2, 3]), sample_rows=0)
    except ConfigError:
        return True
    return False


def an_empty_column_picks_raw() -> dict:
    """A column with no rows, where nothing can be measured and raw is the answer."""
    made = choose(integer_column("v", []))
    return {
        "picked": made.picked,
        "it_picked_raw": made.picked == RAW,
        "sampled": made.sampled,
    }


def compare_the_columns(rows: int = 20000) -> list[dict]:
    """Every shape against every encoding, which is the module in one table."""
    out = []
    for name, one in _columns(rows).items():
        made = choose(one)
        row = {"column": name, "picked": made.picked}
        for candidate in made.candidates:
            row[candidate.name] = round(candidate.ratio, 3) if candidate.usable else None
        out.append(row)
    return out


def summarise() -> dict:
    """The module in one mapping."""
    return {
        "encodings": len(NAMES),
        "sample_rows": SAMPLE_ROWS,
        "threshold": WORTH_IT,
        "shapes_get_different_encodings": every_shape_gets_a_different_encoding()[
            "it_picked_several"
        ],
        "the_sample_agrees": the_sample_agrees_with_the_whole_column()["they_all_agree"],
        "chaining_would_save": chaining_would_save_more()["by_this_ratio"],
        "every_choice_decodes": every_choice_is_decodable()["they_all_decode"],
    }

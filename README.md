# cqe

A vectorised columnar query engine in NumPy: file format, statistics, planner, cost model, SQL
front end, and an execution layer that runs a query end to end.

The thing that makes it unusual is that every module is written as a set of measurements. A
module implements an operator and then, in the same file, measures the claims that operator
rests on. Those functions are what the tests assert on, so a claim in a docstring and a number
in the test suite cannot drift apart. Roughly forty five of those claims turned out to be wrong
when measured, and each correction is recorded in the module that made it rather than quietly
edited out.

```bash
pip install -e ".[dev]"
```

```bash
python -m pytest
```

```bash
python examples/measure_everything.py
```

The last one prints every module's summary in one table. It is the package's own claim about
itself, and a change anywhere in the engine shows up in it.

## What is here

```
cqe/columns     column arrays, null representations, four encodings and a chooser
cqe/types       the logical type system and the casts between types
cqe/exec        filter, project, aggregate, sort, window, distinct, set operations, joins,
                batched pipelines, and spilling
cqe/storage     the file format, row group statistics, zone maps, bloom filters, physical
                layouts, a catalogue, and compaction
cqe/stats       histograms, sketches, cardinality estimation, correlation
cqe/plan        seven logical nodes, four rewrite rules, physical planning, cost attribution
cqe/cost        a meter that counts and a model that predicts
cqe/sql         tokeniser, parser, and a plan to SQL renderer
cqe/verify      a row at a time reference implementation, a fuzzer, a differential harness
cqe/eval        a workload, a regression checker, and fitted scaling exponents
cqe/cli         a command line over all of it
```

## How the cost is counted

Nothing here is timed. Every cost is a count: values touched, bytes read, rows materialised,
hash probes, comparisons. A timing tells you about the machine it ran on and the state of its
caches. A count is a property of the algorithm and the data, so it is reproducible, it can be
asserted on in a test, and a regression in it is a real regression.

`cqe/cost/meter.py` is the counter. `cqe/cost/model.py` predicts the same numbers before the
query runs. `cqe/eval/regression.py` fails when a query's count moves by more than one per cent.

## How correctness is established

Every fast path has a slow one beside it. `cqe/verify/reference.py` implements each operator a
row at a time with no vectorisation and no cleverness, and `cqe/verify/differential.py` runs
both over generated inputs and compares. Where an operator has several strategies, all of them
are checked against each other as well: three join strategies, three aggregate strategies, three
distinct strategies, four null representations.

That harness caught its own worst bug. `Agreement` had no `__bool__`, so a dataclass was always
truthy and every `assert agree(fast, reference)` in the package passed whatever the two sides
held. Adding five lines turned the whole strategy back on, and all 1259 tests passing afterwards
is the evidence that only the check had been broken.

## Some of what the measurements found

Each of these is a claim that was written down, measured, and turned out to be wrong. The
correction lives in the module that made it.

**A sort under a limit makes its own cost negative.** `plan/attribute.py` charges a plan's
measured cost to its nodes by running each subtree and subtracting the children. I assumed no
node could come out below zero. The runner turns a sort under a limit into a partial sort, so
the sort inside the limit does a fifth of the work it does alone, and 31960 values land on the
limit as a negative number. Clamping it would have been wrong: the limit caused the saving.

**The model and the meter disagree about what a scan is, by a factor of thirty two thousand.**
The model prices a scan as bytes off storage. The runner is handed a batch already in memory and
returns it, touching nothing. Both are right about the system they describe, and it is most of
why the model reads as a systematic overestimate.

**Compaction does not trade pruning for metadata.** That is two decisions confused for one.
Merging a hundred and twenty fragments into a file with the same group size prunes identically,
to the row. The loss appears only when the group size changes, which could have been decided at
ingest time. Compaction is what makes a wide group affordable, not what forces one.

**A total that looks right hides nodes that are not.** Across eight plans the root is out by 5.4
on average and the worst node inside it by 101.

**A null row still holds a value underneath it.** Three distinct strategies read that value and
reported it, so a column with two thousand nulls came back with seven separate nulls and a count
that was over by six with nothing looking wrong.

**A bloom filter with two hash functions gives a 6.4 per cent false positive rate.** The measured
optimum at ten bits per entry is seven, matching ln 2 times ten.

**`np.isclose` in a precision loss check destroys the precision it is measuring.**

**Nested loop scaling measures as linear if you only scale one side.**

**A sample of four thousand rows agrees with the whole column on all seven natural shapes** and
is only fooled by a column constructed to have a sorted prefix, which is why that column is in
the tests.

## The command line

```bash
cqe measure
```

```bash
cqe query "select region, count(*) as n from facts group by region" --json
```

`schema`, `stats`, `plan`, `explain`, `cost`, `query`, `write`, `verify`, `measure`.

## Scale

About 30000 lines of implementation and tests, 53 modules, 2555 tests. Python 3.11 or later and
NumPy are the only requirements; there is no other dependency, and no part of the engine calls
out to a database to do the work.

## Attribution

Written by Kiruthika Subramani in collaboration with Claude, Anthropic's AI assistant. The
design, the choice of what to measure, and the review of every result were mine; the
implementation and the tests were written jointly. The measurements above were run rather than
asserted, and about forty five of them contradicted the first version of the claim they now
support.

## Licence

MIT.

# Register-local reduction scheduling benchmark

Run `reduction.py` on a branch supporting `in_thread_reduction`. It compares
`tree` and `sequential` using identical inputs, layouts, launch parameters, and
precision. Only the sum's reduction strategy changes. This is a standalone
benchmark, not a pytest test.

```sh
python python/test/microbenchmark/reduction.py --suite smoke --output /tmp/reduction-smoke
python python/test/microbenchmark/reduction.py --suite full --output /tmp/reduction-full --dump-code
```

Use a fresh output directory for each run. `--workload sum` and
`--workload softmax` run either part separately. `--suite quick` shortens the
sum sweep. `--repeats` and `--rep-ms` control measurement length.
Use `--case-ids 1,2,3` to repeat selected cases from the same suite/workload.

The full suite contains 606 sum cases and 180 softmax cases:

| Sum dimension | Coverage |
|---|---|
| Input / accumulation | FP16 / FP32, BF16 / FP32, FP32 / FP32 |
| Scalar reduction inputs per thread | 1, 2, 4, 8, 16, 32, 64, 128 |
| Independent outputs per thread | 2, 4 |
| Reduction distribution | Registers only, within warps, across warps |
| Register packing | Along reduction axis or across independent outputs |
| Grid size | One CTA or twice the GPU's SM count |
| Additional slices | Both axes; broadcast reduced or retained dimension; 4 or 8 warps |

The additional slices fix the nominal inputs per thread to 16 and outputs to 2.
Broadcasting can reduce the number of distinct values physically held in
registers. The compiler may simplify these cases; generated TTGIR and PTX are
available with `--dump-code`. The one-input cases are controls where changing
the register-local strategy should have no effect. Column reductions retain
row-major physical storage, so their memory access patterns differ from rows.
The raw data flags identical-PTX pairs: timing differences in those controls
must not be attributed to reduction scheduling, especially near the launch floor.

Softmax uses 32 or 4096 rows, widths 128–8192 (including masked widths 1000 and
3000), and 1 or 4 rows per CTA. It uses 4 warps, or 8 for width 8192. All three
input dtypes are covered. FP16/BF16 exponentials are rounded before the FP32
denominator sum, in both policies. This differs from a fully FP32 softmax.
Normalization either reuses the rounded exponentials or retains the original
FP32 exponentials. The former can inhibit the backend's mixed-precision add
matching because the widened values have another use. FP32 input is a control
with just one normalization variant. The maximum reduction always uses its
default strategy.

Each variant is checked against an FP64 reference before measurement. Sum
errors are also scaled by the sum of absolute inputs; softmax reports row-sum
error as well as maximum absolute error against mathematical softmax. Inputs
are normally distributed. These checks are not an exhaustive numerical study.

Timings replay CUDA graphs containing 128 launches, with hot inputs and five
measurement rounds by default. Measurement order alternates between policies.
All timings include loads and stores; these are not isolated instruction
latencies. No autotuning or GPU clock locking is performed. Run on an idle GPU
and repeat interesting cases independently before drawing conclusions.

`results.csv` contains paired median latencies, min/max samples, speedups,
registers, spills, shared-memory usage, static PTX mixed-add counts, and errors.
`results.jsonl` additionally preserves all timing samples and is appended after
each completed case. `metadata.json` records arguments, hardware, software,
the compiler commit, and the benchmark source hash. `summary.md` groups results
and highlights extremes; its 3% thresholds are descriptive, not significance
tests. A speedup above one favors sequential reduction. Regenerate the summary
with `--summarize --output <existing-output-directory>`.

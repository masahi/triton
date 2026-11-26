"""Compare tree and sequential register-local sum scheduling.

Run on the explicit-in-thread-reduction branch, for example:
  python python/test/microbenchmark/reduction.py --suite smoke --output /tmp/reduction-smoke
  python python/test/microbenchmark/reduction.py --output /tmp/reduction-full

The sum suite fixes Gluon layouts to control unique reduction values and
independent outputs per thread. Both policies use the same layout and launch.
Timings replay CUDA graphs with hot inputs; they include loads and stores, not
just arithmetic. Softmax rounds exponentials to the input dtype before its FP32
sum in BOTH variants. This rounding is part of the workload being measured.
Softmax includes rounded and FP32 numerators: reusing widened rounded values
for normalization can prevent the backend's mixed-precision add matching.
"""

import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import re
import statistics
import subprocess

import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl


@gluon.jit
def sum_kernel(X, Y, M: gl.constexpr, N: gl.constexpr, AXIS: gl.constexpr, BROADCAST: gl.constexpr,
               LAYOUT: gl.constexpr, POLICY: gl.constexpr):
    m = gl.arange(0, M, layout=gl.SliceLayout(1, LAYOUT))
    n = gl.arange(0, N, layout=gl.SliceLayout(0, LAYOUT))
    if BROADCAST == "reduced":
        if AXIS == 1:
            x = gl.load(X + gl.program_id(0) * M + m)[:, None]
        else:
            x = gl.load(X + gl.program_id(0) * N + n)[None, :]
        x, _ = gl.broadcast(x, gl.full((M, N), 0, gl.int32, LAYOUT))
    elif BROADCAST == "output":
        if AXIS == 1:
            x = gl.load(X + gl.program_id(0) * N + n)[None, :]
        else:
            x = gl.load(X + gl.program_id(0) * M + m)[:, None]
        x, _ = gl.broadcast(x, gl.full((M, N), 0, gl.int32, LAYOUT))
    else:
        x = gl.load(X + gl.program_id(0) * M * N + m[:, None] * N + n[None, :])
    y = gl.sum(x.to(gl.float32), AXIS, in_thread_reduction=POLICY)
    if AXIS == 1:
        gl.store(Y + gl.program_id(0) * M + m, y)
    else:
        gl.store(Y + gl.program_id(0) * N + n, y)


@gluon.jit
def softmax_kernel(X, Y, ROWS: gl.constexpr, COLS: gl.constexpr, BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr,
                   LAYOUT: gl.constexpr, NUMERATOR: gl.constexpr, POLICY: gl.constexpr):
    m = gl.program_id(0) * BLOCK_M + gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, LAYOUT))
    n = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, LAYOUT))
    mask = (m[:, None] < ROWS) & (n[None, :] < COLS)
    x = gl.load(X + m[:, None] * COLS + n[None, :], mask, other=-float("inf")).to(gl.float32)
    # Only the sum's policy changes. The max keeps its default lowering.
    exp = gl.exp(x - gl.max(x, 1)[:, None])
    p = exp.to(X.dtype.element_ty)
    denom = gl.sum(p.to(gl.float32), 1, in_thread_reduction=POLICY)
    if NUMERATOR == "rounded":
        y = p.to(gl.float32) / denom[:, None]
    else:
        y = exp / denom[:, None]
    gl.store(Y + m[:, None] * COLS + n[None, :], y, mask)


def sum_layout(case):
    """K counts unique scalar reduction inputs per thread before vector packing."""
    warps = case["warps"]
    distribution = case["distribution"]
    spt = [1, min(2, case["k"])] if case["packing"] == "reduction" else [2, 1]
    tpw = [32, 1] if distribution == "register" else [1, 32]
    wpc = [1, warps] if distribution == "cta" else [warps, 1]
    m = case["outputs"] * tpw[0] * wpc[0]
    n = case["k"] * tpw[1] * wpc[1]
    # Avoid extra unique outputs when requesting two-wide output packing.
    if spt[0] == 2:
        m = max(m, 2 * tpw[0] * wpc[0])
    if case["axis"] == 0:
        m, n = n, m
        spt, tpw, wpc = spt[::-1], tpw[::-1], wpc[::-1]
    # Physical memory remains row-major for both axes.
    layout = gl.BlockedLayout(spt, tpw, wpc, [1, 0])
    return m, n, layout


def cases(suite, sm_count):
    dtypes = ("float16", "bfloat16", "float32")
    if suite == "smoke":
        combinations = itertools.product(dtypes, ("register", "warp", "cta"), ("reduction", ), (16, ), (1, ), (4, ),
                                         ("saturated", ))
    else:
        ks = (1, 2, 4, 8, 16, 32, 64, 128) if suite == "full" else (2, 8, 32)
        combinations = itertools.product(dtypes, ("register", "warp", "cta"), ("reduction", "output"), ks, (2, 4),
                                         (4, ), ("small", "saturated"))
    for dtype, distribution, packing, k, outputs, warps, grid in combinations:
        yield dict(workload="sum", dtype=dtype, distribution=distribution, packing=packing, k=k, outputs=outputs,
                   warps=warps, grid=grid, ctas=1 if grid == "small" else 2 * sm_count, axis=1, broadcast="none")
    # Orthogonal slices avoid multiplying every core case by axis/broadcast.
    for dtype, axis, broadcast, warps in itertools.product(dtypes, (0, 1), ("none", "reduced", "output"), (4, 8)):
        if axis == 1 and broadcast == "none":
            continue
        if suite == "smoke" and (dtype != "float16" or warps != 4):
            continue
        yield dict(workload="sum", dtype=dtype, distribution="warp", packing="reduction", k=16, outputs=2, warps=warps,
                   grid="saturated", ctas=2 * sm_count, axis=axis, broadcast=broadcast)
    widths = (128, 1000, 2048, 8192) if suite == "smoke" else (128, 256, 512, 1000, 1024, 2048, 3000, 4096, 8192)
    for dtype, rows, cols, block_m in itertools.product(("float16", "float32") if suite == "smoke" else dtypes,
                                                        (32, 4096), widths, (1, 4)):
        if suite == "smoke" and (rows != 4096 or block_m != 1):
            continue
        for numerator in (("fp32", ) if dtype == "float32" else ("rounded", "fp32")):
            yield dict(workload="softmax", dtype=dtype, rows=rows, cols=cols, block_m=block_m, numerator=numerator,
                       warps=8 if cols == 8192 else 4)


def prepare(case):
    dtype = getattr(torch, case["dtype"])
    if case["workload"] == "sum":
        m, n, layout = sum_layout(case)
        axis = case["axis"]
        shape = [case["ctas"], m, n]
        if case["broadcast"] != "none":
            dim = axis if case["broadcast"] == "reduced" else 1 - axis
            shape[dim + 1] = 1
        x = torch.randn(shape, device="cuda", dtype=dtype)
        values = x.double().expand(case["ctas"], m, n)
        reference = values.sum(axis + 1)
        scale = values.abs().sum(axis + 1)
        case = dict(case, rows=m, cols=n)

        def launch(policy, out):
            return sum_kernel[(case["ctas"], )](x, out, m, n, axis, case["broadcast"], layout, policy,
                                                num_warps=case["warps"])
    else:
        rows, cols = case["rows"], case["cols"]
        x = torch.randn((rows, cols), device="cuda", dtype=dtype)
        # Mathematical softmax reference: also exposes error from rounded exp.
        reference = torch.softmax(x.double(), dim=1)
        scale = None
        block_m = case["block_m"]
        # One row per warp when BLOCK_M=4; otherwise one row spans the CTA.
        layout = gl.BlockedLayout([1, 2], [1, 32], [block_m, case["warps"] // block_m], [1, 0])

        def launch(policy, out):
            return softmax_kernel[(triton.cdiv(rows, block_m), )](x, out, rows, cols, block_m,
                                                                  triton.next_power_of_2(cols), layout,
                                                                  case["numerator"], policy, num_warps=case["warps"])

    shape = reference.shape
    output_dtype = torch.float32 if case["workload"] == "sum" else dtype
    # Use identical pointer arguments as well as identical launch parameters.
    out = torch.empty(shape, device="cuda", dtype=output_dtype)
    functions = {p: (lambda p=p: launch(p, out)) for p in ("tree", "sequential")}
    compiled = {}
    errors = {}
    for policy, fn in functions.items():
        compiled[policy] = fn()
        error = (out.double() - reference).abs()
        if scale is not None:
            torch.testing.assert_close(error <= 1e-4 * scale + 1e-6, torch.ones_like(error, dtype=torch.bool))
            scaled_error = (error / scale.clamp_min(1e-30)).max().item()
        else:
            torch.testing.assert_close(out.double(), reference, rtol=0.02 if dtype == torch.bfloat16 else 0.003,
                                       atol=1e-7)
            scaled_error = ((out.double().sum(1) - 1).abs()).max().item()
        errors[policy] = dict(max_abs_error=error.max().item(), scaled_error=scaled_error)
    return case, functions, compiled, errors


def benchmark_pair(functions, repeats, rep_ms):
    graphs = {}
    for policy, fn in functions.items():
        for _ in range(8):
            fn()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(128):
                fn()
        graphs[policy] = graph
    samples = {p: [] for p in functions}
    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    # Estimate replay length after warming BOTH variants.
    replays = {}
    for policy, graph in graphs.items():
        graph.replay()
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        replays[policy] = max(1, math.ceil(rep_ms / start.elapsed_time(end)))
    for repeat in range(repeats):
        order = ("tree", "sequential") if repeat % 2 == 0 else ("sequential", "tree")
        for policy in order:
            start.record()
            for _ in range(replays[policy]):
                graphs[policy].replay()
            end.record()
            end.synchronize()
            samples[policy].append(start.elapsed_time(end) * 1000 / (128 * replays[policy]))
    return samples


def write_summary(records, directory):
    groups = {}
    for record in records:
        if record["workload"] == "sum":
            precision = {"float16": "FP16→FP32", "bfloat16": "BF16→FP32", "float32": "FP32→FP32"}[record["dtype"]]
            group = f"sum / {precision} / {record['grid']}"
        else:
            group = f"softmax / {record['dtype']} / {record['numerator']} numerator / {record['rows']} rows"
        groups.setdefault(group, []).append(record)
    lines = [
        "# Reduction scheduling results", "",
        "Latency reduction (%) = 100 × (tree − sequential) / tree; positive favors sequential.",
        "Aggregate ratios are tree latency / sequential latency (×); above 1 favors sequential.",
        "Timings include memory traffic. All cases passed numerical checks against FP64 references.",
        "Ratio thresholds below are descriptive, not statistical significance tests.", "",
        "| Workload | Cases | Geomean ratio (×) | Min ratio (×) | Max ratio (×) | Ratio >1.03× | Ratio <1/1.03× |",
        "|---|---:|---:|---:|---:|---:|---:|"
    ]
    for group, entries in sorted(groups.items()):
        ratios = [r["speedup"] for r in entries]
        mean = math.exp(statistics.mean(math.log(r) for r in ratios))
        lines.append(f"| {group} | {len(entries)} | {mean:.3f} | {min(ratios):.3f} | {max(ratios):.3f} | "
                     f"{sum(r > 1.03 for r in ratios)} | {sum(r < 1 / 1.03 for r in ratios)} |")
    for workload in ("sum", "softmax"):
        entries = sorted((r for r in records if r["workload"] == workload), key=lambda r: r["speedup"])
        if not entries:
            continue
        lines.extend([
            "", f"## {workload}: extremes", "",
            "Use the raw samples and reruns in separate benchmark invocations to assess stability of individual cases.",
            "",
            "| Case | Tree (µs) | Sequential (µs) | Latency reduction (%) | Mixed adds tree/seq | Registers tree/seq |",
            "|---|---:|---:|---:|---:|---:|"
        ])
        for record in entries[:3] + entries[-3:]:
            description = {
                k: v
                for k, v in record.items()
                if k in ("case_id", "dtype", "rows", "cols", "block_m", "numerator", "distribution", "packing", "grid",
                         "k", "outputs", "axis", "broadcast", "warps")
            }
            lines.append(
                f"| {description} | {record['tree_us']:.3f} | {record['sequential_us']:.3f} | "
                f"{100 * (1 - 1 / record['speedup']):+.1f}% | {record['tree_mixed_adds']}/{record['sequential_mixed_adds']} | "
                f"{record['tree_registers']}/{record['sequential_registers']} |")
    (directory / "summary.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--suite", choices=("smoke", "quick", "full"), default="full")
    parser.add_argument("--workload", choices=("all", "sum", "softmax"), default="all")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--rep-ms", type=float, default=10)
    parser.add_argument("--dump-code", action="store_true", help="Save PTX and TTGIR for every variant")
    parser.add_argument("--summarize", action="store_true", help="Regenerate summary.md from existing results.jsonl")
    parser.add_argument("--case-ids", help="Comma-separated case IDs within the selected suite/workload")
    args = parser.parse_args()
    if args.summarize:
        records = [json.loads(line) for line in (args.output / "results.jsonl").read_text().splitlines()]
        write_summary(records, args.output)
        return
    if args.repeats < 2 or args.rep_ms <= 0:
        parser.error("--repeats must be >= 2 and --rep-ms must be positive")
    if (args.output / "results.jsonl").exists():
        parser.error("Choose a fresh output directory; results.jsonl already exists")
    if triton.runtime.driver.active.get_current_target().backend != "cuda":
        parser.error("This benchmark requires an NVIDIA GPU")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(17)
    props = torch.cuda.get_device_properties(0)
    metadata = dict(arguments={**vars(args), "output":
                               str(args.output)}, gpu=props.name, capability=torch.cuda.get_device_capability(),
                    sm_count=props.multi_processor_count, torch=torch.__version__, cuda=torch.version.cuda,
                    triton=triton.__version__, git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"],
                                                                                  text=True).strip(),
                    benchmark_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    timing="Hot-input CUDA graphs, 128 launches per replay; alternating policy order")
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    selected = [
        c for c in cases(args.suite, props.multi_processor_count)
        if args.workload == "all" or c["workload"] == args.workload
    ]
    indexed = list(enumerate(selected))
    if args.case_ids:
        requested = {int(i) for i in args.case_ids.split(",")}
        if requested - set(range(len(selected))):
            parser.error("--case-ids contains IDs outside the selected suite/workload")
        indexed = [(i, c) for i, c in indexed if i in requested]
    records = []
    for completed, (index, case) in enumerate(indexed, 1):
        case, functions, compiled, errors = prepare(case)
        samples = benchmark_pair(functions, args.repeats, args.rep_ms)
        medians = {p: statistics.median(s) for p, s in samples.items()}
        record = dict(case_id=index, **case, speedup=medians["tree"] / medians["sequential"],
                      identical_ptx=compiled["tree"].asm["ptx"] == compiled["sequential"].asm["ptx"])
        for policy, kernel in compiled.items():
            record.update({f"{policy}_{key}": value for key, value in errors[policy].items()})
            record.update({
                f"{policy}_us":
                medians[policy], f"{policy}_min_us":
                min(samples[policy]), f"{policy}_max_us":
                max(samples[policy]), f"{policy}_registers":
                kernel.n_regs, f"{policy}_spills":
                kernel.n_spills, f"{policy}_shared_bytes":
                kernel.metadata.shared, f"{policy}_mixed_adds":
                len(re.findall(r"\badd\.rn\.f32x2\.(?:f16|bf16)x2\.f32x2\b", kernel.asm["ptx"]))
            })
            if args.dump_code:
                for kind in ("ptx", "ttgir"):
                    (args.output / f"{index:04d}-{policy}.{kind}").write_text(kernel.asm[kind])
        record["samples_us"] = samples
        records.append(record)
        # Append immediately so interrupted runs retain completed cases.
        with (args.output / "results.jsonl").open("a") as output:
            output.write(json.dumps(record) + "\n")
        print(
            f"[{completed}/{len(indexed)}] case={index} {case} tree={medians['tree']:.3f}us "
            f"sequential={medians['sequential']:.3f}us speedup={record['speedup']:.3f}", flush=True)
    keys = sorted({k for r in records for k in r if k != "samples_us"})
    with (args.output / "results.csv").open("w") as output:
        writer = csv.DictWriter(output, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    write_summary(records, args.output)


if __name__ == "__main__":
    main()

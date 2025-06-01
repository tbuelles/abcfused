import contextlib
import gc
import inspect
import time
from typing import Set
import pandas as pd

import torch
import torch.profiler
import triton.language as tl

TORCH_TO_TL_DTYPE = {
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
    torch.float32: tl.float32,
    torch.int32: tl.int32,
    torch.int64: tl.int64,
    torch.bool: tl.int1,  # Triton doesn't have bool, uses int1
}


def is_power_of_2(x):
    return (x & (x - 1)) == 0 and x > 0


def pprint(*args, **kwargs):
    print("-" * 100)
    print(*args, **kwargs)


def filter_kwargs(obj, kwargs, *, keep_keys=[]):
    if callable(obj):
        sig = inspect.signature(obj)
        params = sig.parameters
        has_var_kwargs = any(p.kind == p.VAR_KEYWORD for p in params.values())
        valid_keys = set(params.keys())
    elif isinstance(obj, (set, list, tuple)):
        has_var_kwargs = False
        valid_keys = set(obj)
    elif isinstance(obj, dict):
        has_var_kwargs = False
        valid_keys = set(obj.keys())
    else:
        raise TypeError(f"Unsupported type for `obj`: {type(obj)}")

    filtered_kwargs = {}
    for k in list(kwargs.keys()):
        if has_var_kwargs or k in valid_keys or k in keep_keys:
            filtered_kwargs[k] = kwargs[k]

    gc.collect()
    torch.cuda.empty_cache()
    return filtered_kwargs


@contextlib.contextmanager
def stats(label="Stats"):
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    start_time = time.perf_counter()
    start_memory = torch.cuda.memory_allocated()

    try:
        yield
    finally:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        end_time = time.perf_counter()
        end_memory = torch.cuda.max_memory_allocated()
        total_memory = (end_memory - start_memory) / 1e6
        total_time = (end_time - start_time) * 1000  # in ms

        print("-" * 100)
        print(f"{label}")
        print(f"{'Total Time:':<15}{total_time:>10.3f} ms")
        print(f"{'Peak Memory:':<15}{total_memory:>10.2f} MB")
        print("-" * 100)


def stats_fn(
    fn,
    inputs,
    gradients,
    /,
    *,
    verbose=True,
    label="",
    n_warmup=None,
    n_repeat=None,
    max_ms=None,
    flush_cache=True,
):
    def _grad_to_none(inputs):
        for arg in inputs.values():
            if isinstance(arg, torch.Tensor) and arg.requires_grad:
                arg.grad = None

    def _bwd(outputs, gradients):
        assert isinstance(outputs, torch.Tensor) and isinstance(gradients, torch.Tensor)
        assert outputs.is_contiguous() and gradients.is_contiguous()
        outputs.backward(gradients)

    def _fwd(func, inputs):
        return func(**inputs)

    # Inputs
    inputs = filter_kwargs(fn, inputs)

    # Stream
    stream = torch.cuda.current_stream()

    # Infer device
    device = next(
        (t.device for t in list(inputs.values()) if isinstance(t, torch.Tensor)),
        torch.device("cuda"),
    )

    # Gradients
    requires_grad_inputs = any(
        isinstance(v, torch.Tensor) and v.requires_grad for v in inputs.values()
    )
    backward = gradients is not None
    if backward and not requires_grad_inputs:
        print("[warning] backward requested, but no input requires gradients.")

    # Cache
    if flush_cache:
        # allocating 256MB to make sure no input data in L2 cache
        cache = torch.empty(int(256e6), dtype=torch.int8, device=device)
    else:
        cache = torch.empty(0, dtype=torch.int8, device=device)

    # Estimate
    if isinstance(max_ms, (int, float)):
        assert n_warmup is None and n_repeat is None, (
            f"if max_ms is provided, n_warmup {n_warmup} and n_repeat {n_warmup} should be None"
        )
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record(stream)
        for _ in range(5):
            cache.zero_()
            _grad_to_none(inputs)
            outputs = _fwd(fn, inputs)
            if backward:
                _bwd(outputs, gradients)
        end_event.record(stream)
        torch.cuda.synchronize()
        estimate_ms = start_event.elapsed_time(end_event) / 5

        n_warmup = max(1, int(0.2 * max_ms / estimate_ms))
        n_repeat = max(1, int(0.8 * max_ms / estimate_ms))

    # Warmup
    assert isinstance(n_warmup, int) and isinstance(n_repeat, int)
    torch.cuda.empty_cache()
    for _ in range(n_warmup):
        cache.zero_()
        _grad_to_none(inputs)
        outputs = _fwd(fn, inputs)
        if backward:
            cache.zero_()
            _bwd(outputs, gradients)

    # Reset memory stats
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device=device)
    torch.cuda.synchronize()

    # Measure
    fwd_time = 0.0
    bwd_time = 0.0
    fwd_start_event = [torch.cuda.Event(enable_timing=True) for _ in range(n_repeat)]
    bwd_start_event = [torch.cuda.Event(enable_timing=True) for _ in range(n_repeat)]
    fwd_end_event = [torch.cuda.Event(enable_timing=True) for _ in range(n_repeat)]
    bwd_end_event = [torch.cuda.Event(enable_timing=True) for _ in range(n_repeat)]

    for it in range(n_repeat):
        cache.zero_()
        _grad_to_none(inputs)
        torch.cuda._sleep(10_000_000)
        fwd_start_event[it].record(stream)
        outputs = _fwd(fn, inputs)
        fwd_end_event[it].record(stream)
        torch.cuda.synchronize()
        fwd_time += fwd_start_event[it].elapsed_time(fwd_end_event[it])

        if backward and requires_grad_inputs:
            cache.zero_()
            torch.cuda._sleep(10_000_000)
            bwd_start_event[it].record(stream)
            _bwd(outputs, gradients)
            bwd_end_event[it].record(stream)
            torch.cuda.synchronize()
            bwd_time += bwd_start_event[it].elapsed_time(bwd_end_event[it])

    # Final stats
    torch.cuda.synchronize()
    fwd_time = fwd_time / n_repeat
    bwd_time = bwd_time / n_repeat
    total_time = fwd_time + bwd_time if backward else fwd_time
    peak_memory_alloc = torch.cuda.max_memory_allocated() / 1e6
    peak_memory_reserved = torch.cuda.max_memory_reserved() / 1e6

    # Print
    if verbose:
        # Prepare all lines first
        lines = [
            f"| {label} |",
            f"{'Peak memory alloc:':<30}{peak_memory_alloc:>10.2f} MB",
            f"{'Peak memory reserved:':<30}{peak_memory_reserved:>10.2f} MB",
            f"{'Forward:':<30}{fwd_time:>10.2f} ms"
        ]

        if backward:
            lines.append(f"{'Backward:':<30}{bwd_time:>10.2f} ms")

        lines.append(f"{'Total time:':<30}{total_time:>10.2f} ms")

        # Compute the max line width
        line_width = max(len(line) for line in lines)

        # Print the result
        print("-" * line_width)
        for line in lines:
            print(line)
        print("-" * line_width)


    return {
        "fn": fn,
        "o": outputs,
        "total_time": total_time,
        "fwd_time": fwd_time,
        "bwd_time": bwd_time,
        "peak_memory_alloc": peak_memory_alloc,
        "peak_memory_reserved": peak_memory_reserved,
    }


def torch_profiler(
    name,
    fn,
    *args,
    verbose=False,
    sort_by="cuda_time_total",
    row_limit=5,
    save_trace=False,
    device="cuda:0",
    **kwargs,
):
    # Warmup
    for _ in range(10):
        out = fn(*args, **kwargs)
        out.sum().backward()

    for arg in args:
        if isinstance(arg, torch.Tensor) and arg.requires_grad and arg.grad is not None:
            arg.grad.zero_()

    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    # CUDA timing and memory
    start_alloc = torch.cuda.memory_allocated(device)
    fwd_start = torch.cuda.Event(enable_timing=True)
    fwd_end = torch.cuda.Event(enable_timing=True)
    bwd_start = torch.cuda.Event(enable_timing=True)
    bwd_end = torch.cuda.Event(enable_timing=True)

    # Start profiling
    on_trace_ready = (
        torch.profiler.tensorboard_trace_handler(f"./log/{name}_trace")
        if save_trace
        else None
    )
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory=True,
        on_trace_ready=on_trace_ready,
        with_stack=True,
    ) as prof:
        # Forward
        fwd_start.record()  # type: ignore
        out = fn(*args, **kwargs)
        fwd_end.record()  # type: ignore

        # Backward
        bwd_start.record()  # type: ignore
        out.sum().backward()
        bwd_end.record()  # type: ignore

        # Step
        prof.step()

    # Wait for events to complete
    torch.cuda.synchronize()

    peak_alloc = torch.cuda.max_memory_allocated(device)
    fwd_time_ms = fwd_start.elapsed_time(fwd_end)
    bwd_time_ms = bwd_start.elapsed_time(bwd_end)
    used_alloc = (peak_alloc - start_alloc) / 1e6

    # Summary Table
    if verbose:
        events = prof.key_averages(group_by_input_shape=True)
        print(f"\nProfiling Summary: {name} (sorted by {sort_by})")
        print(events.table(sort_by=sort_by, row_limit=row_limit))

    # CUDA Event timing summary
    line_width = max(29, len(name)) + 4
    print("-" * line_width)
    print(f"{name}")
    print(f"{'Forward:':<15}{fwd_time_ms:>10.3f} ms")
    print(f"{'Backward:':<15}{bwd_time_ms:>10.3f} ms")
    print(f"{'Total:':<15}{fwd_time_ms + bwd_time_ms:>10.3f} ms")
    print(f"{'Peak Memory:':<15}{used_alloc:>10.2f} MB")
    print("-" * line_width)


def assert_allclose(x, y, label=None, atol=1e-3):
    if x is None or y is None:
        return

    assert x.shape == y.shape, f"`x.shape` {x.shape} `y.shape` {y.shape}"

    mask = torch.isfinite(x)
    mask2 = torch.isfinite(y)

    print("-" * 100)
    if label is not None:
        print(f"| {label} |")
        print("-" * (len(label) + 4))
        print(
            f"x NaN: {(~mask).sum().item()}, x max: {x[mask].max().item():.4f}, x mean: {x[mask].mean().item():.4f}"
        )
        print(
            f"y NaN: {(~mask2).sum().item()}, y max: {y[mask2].max().item():.4f}, y mean: {y[mask2].mean().item():.4f}"
        )

    mask = mask & mask2
    diff = (x - y).abs()
    max_diff = diff[mask].max().item() if mask.any() else float("nan")
    mean_diff = diff[mask].mean().item() if mask.any() else float("nan")
    print(f"diff max: {max_diff:.1e}, diff mean: {mean_diff:.1e}")
    if max_diff > atol:
        print("-" * 100)
        print("-" * 45, " FAILED ", "-" * 45)
        print("-" * 100)


def assert_dict_allclose(d1, d2, rtol=1e-5, atol=1e-8):
    """
    Compare two dictionaries for equality. Values may be strings, floats, or torch.Tensors.

    - Uses `torch.allclose` for tensor comparison.
    - Uses `==` for strings and floats.

    Raises:
        AssertionError: If any key is missing or mismatched.
    """
    assert d1.keys() == d2.keys(), f"Keys mismatch: {d1.keys()} vs {d2.keys()}"

    for k in d1:
        v1, v2 = d1[k], d2[k]

        if isinstance(v1, torch.Tensor) and isinstance(v2, torch.Tensor):
            assert torch.allclose(v1, v2, rtol=rtol, atol=atol), (
                f"Tensor mismatch at key '{k}'"
            )
        elif isinstance(v1, (float, int)) and isinstance(v2, (float, int)):
            assert abs(v1 - v2) <= atol + rtol * abs(v2), (
                f"Float mismatch at key '{k}': {v1} vs {v2}"
            )
        else:
            assert v1 == v2, f"Value mismatch at key '{k}': {v1} vs {v2}"

    pprint("Success, all tests pass.")


def get_fn_keys(fn) -> Set[str]:
    sig = inspect.signature(fn)
    return set(sig.parameters.keys())

def profiler_to_dataframe(prof) -> pd.DataFrame:
    # Collect raw events
    records = []
    for event in prof.events():
        records.append({
            "name": event.name,
            "cpu_time": event.cpu_time_total,
            "device_time": event.device_time_total,
        })

    # Create DataFrame
    df = pd.DataFrame(records)

    # Group by kernel name and aggregate
    grouped = (
        df.groupby("name", as_index=False)
          .agg(
              cpu_time_total=("cpu_time", "sum"),
              device_time_total=("device_time", "sum"),
              calls=("name", "count")
          )
    )

    # Sort by device time descending
    grouped = grouped.sort_values(by="device_time_total", ascending=False)  # type: ignore

    # Convert to ms and format with "ms" suffix
    grouped["cpu_time_total"] = (grouped["cpu_time_total"] / 1e3).apply(lambda x: f"{x:.2f} ms")
    grouped["device_time_total"] = (grouped["device_time_total"] / 1e3).apply(lambda x: f"{x:.2f} ms")

    return grouped.reset_index(drop=True)

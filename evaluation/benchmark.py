"""Model benchmarking: throughput, latency, memory, FLOPs, parameter count."""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn
import torch.utils.benchmark as benchmark

from calflops import calculate_flops

logger = logging.getLogger(__name__)


def count_parameters(model: nn.Module) -> dict[str, int]:
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total_params": total, "trainable_params": trainable}


def compute_flops(
    model: nn.Module,
    input_size: tuple[int, ...],
) -> dict[str, str]:
    """Compute FLOPs and MACs using calflops.

    Returns
    -------
    dict with keys: flops, macs, calflops_params
    """
    flops, macs, params = calculate_flops(
        model=model,
        input_shape=input_size,
        output_as_string=True,
        output_precision=4,
        print_results=False,
        print_detailed=False,
    )
    return {"flops": flops, "macs": macs, "calflops_params": params}


@torch.no_grad()
def measure_throughput(
    model: nn.Module,
    input_size: tuple[int, ...],
    device: torch.device,
    warmup_runs: int = 5,
    timed_runs: int = 20,
) -> dict[str, float]:
    """Measure inference throughput and latency using torch.utils.benchmark.

    Returns
    -------
    dict with keys: mean_latency_ms, median_latency_ms, iqr_latency_ms,
                    throughput_samples_per_sec
    """
    model.eval().to(device)
    dummy = torch.randn(input_size, device=device)

    timer = benchmark.Timer(
        stmt="model(x)",
        globals={"model": model, "x": dummy},
        num_threads=1,
        label="Inference",
        sub_label=str(input_size),
        description="forward",
    )

    measurement = timer.timeit(number=timed_runs)

    batch_size = input_size[0]
    mean_lat_ms = measurement.mean * 1000
    median_lat_ms = measurement.median * 1000
    iqr_lat_ms = measurement.iqr * 1000

    return {
        "mean_latency_ms": mean_lat_ms,
        "median_latency_ms": median_lat_ms,
        "iqr_latency_ms": iqr_lat_ms,
        "throughput_samples_per_sec": batch_size / (mean_lat_ms / 1000),
    }


def measure_gpu_memory(
    model: nn.Module,
    input_size: tuple[int, ...],
    device: torch.device,
) -> dict[str, float]:
    """Measure peak GPU memory during a forward pass (CUDA only).

    Returns memory values in MB.
    """
    if device.type != "cuda":
        return {"peak_memory_mb": float("nan")}

    model.eval().to(device)
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()

    dummy = torch.randn(input_size, device=device)
    with torch.no_grad():
        _ = model(dummy)
    torch.cuda.synchronize()

    peak = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    return {"peak_memory_mb": peak}


def run_benchmark(
    model: nn.Module,
    cfg: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    """Run full benchmark suite and return a results dict."""
    bcfg = cfg["benchmark"]
    input_size = tuple(bcfg["input_size"])

    results: dict[str, Any] = {}
    results.update(count_parameters(model))
    results.update(compute_flops(model, input_size))
    results.update(
        measure_throughput(
            model, input_size, device,
            warmup_runs=bcfg["warmup_runs"],
            timed_runs=bcfg["timed_runs"],
        )
    )
    results.update(measure_gpu_memory(model, input_size, device))

    return results

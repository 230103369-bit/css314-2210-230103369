import os

# Keep NumPy operations single-threaded.
# Parallelism is controlled by our explicit worker pool.
for variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
):
    os.environ[variable] = "1"

import statistics
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np


STUDENT_ID = 230103369
ARRAY_SIZE = 30_000_000
REPETITIONS = 5


def memory_kernel(a, b, start, stop):
    """Kernel A: memory-bound vector copy."""
    np.copyto(
        b[start:stop],
        a[start:stop],
    )


def compute_kernel(a, b, start, stop):
    """Kernel B: compute-bound trigonometric expression."""
    a_chunk = a[start:stop]
    b_chunk = b[start:stop]

    np.multiply(
        np.sin(a_chunk),
        np.cos(a_chunk),
        out=b_chunk,
    )

    b_chunk += np.tan(a_chunk)


def run_parallel_kernel(
    kernel,
    a,
    b,
    workers,
):
    """Run one kernel using the requested worker count."""

    chunk_size = (
        a.size + workers - 1
    ) // workers

    with ThreadPoolExecutor(
        max_workers=workers
    ) as executor:

        futures = []

        for start in range(
            0,
            a.size,
            chunk_size,
        ):
            stop = min(
                start + chunk_size,
                a.size,
            )

            futures.append(
                executor.submit(
                    kernel,
                    a,
                    b,
                    start,
                    stop,
                )
            )

        for future in futures:
            future.result()


def benchmark_kernel(
    kernel,
    a,
    b,
    workers,
):
    """Return median wall-clock time in milliseconds."""

    times = []

    # Warm-up.
    run_parallel_kernel(
        kernel,
        a,
        b,
        workers,
    )

    for _ in range(REPETITIONS):

        start = time.perf_counter()

        run_parallel_kernel(
            kernel,
            a,
            b,
            workers,
        )

        end = time.perf_counter()

        times.append(
            (end - start) * 1000
        )

    return statistics.median(times)


def benchmark_all(
    kernel,
    a,
    b,
    worker_counts,
):
    """Benchmark one kernel for every requested core count."""

    results = []

    baseline_time = None

    for workers in worker_counts:

        wall_time = benchmark_kernel(
            kernel,
            a,
            b,
            workers,
        )

        if baseline_time is None:
            baseline_time = wall_time

        speedup = (
            baseline_time / wall_time
        )

        efficiency = (
            speedup / workers
        )

        results.append(
            {
                "workers": workers,
                "time": wall_time,
                "speedup": speedup,
                "efficiency": efficiency,
            }
        )

    return results


def print_results(
    title,
    results,
):
    print(title)

    print(
        "Cores | Wall Time (ms) | "
        "Speedup | Efficiency"
    )

    print(
        "------|----------------|"
        "---------|-----------"
    )

    for result in results:

        print(
            f"{result['workers']:5d} | "
            f"{result['time']:14.3f} | "
            f"{result['speedup']:7.3f} | "
            f"{result['efficiency']:9.3f}"
        )

    print()


def main():

    max_cores = os.cpu_count() or 1

    worker_counts = [
        1,
        2,
        max_cores,
    ]

    print(
        f"Student ID: {STUDENT_ID}"
    )

    print(
        f"Array size: {ARRAY_SIZE}"
    )

    print(
        "Data type: float64"
    )

    print(
        f"Available logical cores: "
        f"{max_cores}"
    )

    print(
        f"Worker counts: "
        f"{worker_counts}"
    )

    print(
        f"Timing statistic: median of "
        f"{REPETITIONS} runs"
    )

    print()

    rng = np.random.default_rng(
        STUDENT_ID
    )

    # Values are kept in a safe range for tan().
    a = rng.random(
        ARRAY_SIZE,
        dtype=np.float64,
    )

    b = np.empty_like(a)

    memory_results = benchmark_all(
        memory_kernel,
        a,
        b,
        worker_counts,
    )

    # Verify Kernel A.
    if not np.array_equal(
        a,
        b,
    ):
        raise RuntimeError(
            "Kernel A verification failed."
        )

    compute_results = benchmark_all(
        compute_kernel,
        a,
        b,
        worker_counts,
    )

    # Verify a sample from Kernel B without allocating
    # another full 30-million-element result array.
    sample_indices = np.array(
        [
            0,
            ARRAY_SIZE // 2,
            ARRAY_SIZE - 1,
        ]
    )

    expected_sample = (
        np.sin(a[sample_indices])
        * np.cos(a[sample_indices])
        + np.tan(a[sample_indices])
    )

    if not np.allclose(
        b[sample_indices],
        expected_sample,
        rtol=1e-12,
        atol=1e-12,
    ):
        raise RuntimeError(
            "Kernel B verification failed."
        )

    print_results(
        "Kernel A - Memory-Bound Vector Copy",
        memory_results,
    )

    print_results(
        "Kernel B - Compute-Bound Trigonometric",
        compute_results,
    )

    memory_max = memory_results[-1]
    compute_max = compute_results[-1]

    print("Scaling Analysis")
    print("-" * 55)

    print(
        f"Kernel A efficiency at "
        f"{max_cores} cores: "
        f"{memory_max['efficiency']:.3f}"
    )

    print(
        f"Kernel B efficiency at "
        f"{max_cores} cores: "
        f"{compute_max['efficiency']:.3f}"
    )

    if (
        memory_max["efficiency"]
        < compute_max["efficiency"]
    ):
        stalled_kernel = (
            "Kernel A (Memory-Bound)"
        )
    else:
        stalled_kernel = (
            "Kernel B (Compute-Bound)"
        )

    print()
    print(
        "Memory channel saturation: "
        f"{stalled_kernel} stalled more "
        "in scaling."
    )

    print()
    print(
        "Result verification: PASSED"
    )


if __name__ == "__main__":
    main()
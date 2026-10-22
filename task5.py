"""Task 5: find where parallel vector work overtakes serial execution."""

import os

# Keep NumPy operations single-threaded.
# Parallelism is controlled only by our explicit ThreadPoolExecutor.
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
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np


STUDENT_ID = 230103369

ARRAY_SIZES = [
    100,
    1_000,
    10_000,
    100_000,
    1_000_000,
    10_000_000,
]

A = 2.5
B = 1.25

REPETITIONS = 21


def serial_operation(x, y):
    """Compute y[i] = A * x[i] + B serially."""
    np.multiply(x, A, out=y)
    np.add(y, B, out=y)


def vector_chunk(x, y, start, stop):
    """Compute one non-overlapping part of the vector."""
    x_chunk = x[start:stop]
    y_chunk = y[start:stop]

    np.multiply(x_chunk, A, out=y_chunk)
    np.add(y_chunk, B, out=y_chunk)


def parallel_operation(executor, x, y, worker_count):
    """Compute the vector operation using all available workers."""

    chunk_size = (
        x.size + worker_count - 1
    ) // worker_count

    futures = []

    for start in range(
        0,
        x.size,
        chunk_size,
    ):
        stop = min(
            start + chunk_size,
            x.size,
        )

        future = executor.submit(
            vector_chunk,
            x,
            y,
            start,
            stop,
        )

        futures.append(future)

    for future in futures:
        future.result()


def elapsed_ms(operation, *arguments):
    """Measure wall-clock execution time in milliseconds."""

    start = time.perf_counter()

    operation(*arguments)

    end = time.perf_counter()

    return (end - start) * 1000


def start_all_workers(executor, worker_count):
    """
    Start every worker before benchmark timing begins.

    This keeps thread creation outside the measured vector operation.
    """

    barrier = threading.Barrier(
        worker_count + 1
    )

    futures = [
        executor.submit(barrier.wait)
        for _ in range(worker_count)
    ]

    barrier.wait()

    for future in futures:
        future.result()


def benchmark_size(
    size,
    executor,
    worker_count,
):
    """Benchmark serial and parallel execution for one N."""

    rng = np.random.default_rng(
        STUDENT_ID + size
    )

    x = rng.random(
        size,
        dtype=np.float64,
    )

    serial_y = np.empty_like(x)
    parallel_y = np.empty_like(x)

    # Warm both paths before measured runs.
    serial_operation(
        x,
        serial_y,
    )

    parallel_operation(
        executor,
        x,
        parallel_y,
        worker_count,
    )

    serial_times = []
    parallel_times = []

    # Alternate execution order to reduce ordering bias.
    for repetition in range(REPETITIONS):

        if repetition % 2 == 0:
            serial_time = elapsed_ms(
                serial_operation,
                x,
                serial_y,
            )

            parallel_time = elapsed_ms(
                parallel_operation,
                executor,
                x,
                parallel_y,
                worker_count,
            )

        else:
            parallel_time = elapsed_ms(
                parallel_operation,
                executor,
                x,
                parallel_y,
                worker_count,
            )

            serial_time = elapsed_ms(
                serial_operation,
                x,
                serial_y,
            )

        serial_times.append(
            serial_time
        )

        parallel_times.append(
            parallel_time
        )

    # Verify that serial and parallel calculations agree.
    if not np.allclose(
        serial_y,
        parallel_y,
        rtol=1e-12,
        atol=1e-12,
    ):
        raise RuntimeError(
            f"Result verification failed for N={size}"
        )

    serial_median = statistics.median(
        serial_times
    )

    parallel_median = statistics.median(
        parallel_times
    )

    return (
        serial_median,
        parallel_median,
    )


def find_crossovers(results):
    """
    Find every tested range where parallel changes
    from slower than serial to faster than serial.
    """

    crossovers = []

    for previous, current in zip(
        results,
        results[1:],
    ):

        previous_parallel_slower = (
            previous["parallel"]
            >= previous["serial"]
        )

        current_parallel_faster = (
            current["parallel"]
            < current["serial"]
        )

        if (
            previous_parallel_slower
            and current_parallel_faster
        ):
            crossovers.append(
                (
                    previous["n"],
                    current["n"],
                )
            )

    return crossovers


def main():

    worker_count = os.cpu_count() or 1

    print(
        f"Student ID: {STUDENT_ID}"
    )

    print(
        f"Available logical cores / workers: "
        f"{worker_count}"
    )

    print(
        f"Operation: y[i] = "
        f"{A} * x[i] + {B}"
    )

    print(
        f"Timing statistic: median of "
        f"{REPETITIONS} runs"
    )

    print()

    results = []

    # One persistent thread pool is used for the complete benchmark.
    with ThreadPoolExecutor(
        max_workers=worker_count
    ) as executor:

        # Create all workers before timing begins.
        start_all_workers(
            executor,
            worker_count,
        )

        for size in ARRAY_SIZES:

            serial_time, parallel_time = (
                benchmark_size(
                    size,
                    executor,
                    worker_count,
                )
            )

            speedup = (
                serial_time / parallel_time
            )

            if parallel_time < serial_time:
                winner = "Parallel"
            else:
                winner = "Serial"

            results.append(
                {
                    "n": size,
                    "serial": serial_time,
                    "parallel": parallel_time,
                    "speedup": speedup,
                    "winner": winner,
                }
            )

    print(
        "N          | Serial (ms) | "
        "Parallel (ms) | Speedup | Faster"
    )

    print(
        "-----------|-------------|"
        "---------------|---------|---------"
    )

    for result in results:

        print(
            f"{result['n']:10d} | "
            f"{result['serial']:11.3f} | "
            f"{result['parallel']:13.3f} | "
            f"{result['speedup']:7.3f} | "
            f"{result['winner']}"
        )

    print()

    crossovers = find_crossovers(
        results
    )

    if crossovers:

        for lower, upper in crossovers:

            print(
                "Measured slower-to-faster "
                "crossover range: "
                f"{lower:,} < N <= {upper:,}"
            )

    elif all(
        result["winner"] == "Parallel"
        for result in results
    ):

        print(
            "Parallel was already faster at "
            "the smallest tested size; "
            f"crossover is at "
            f"N <= {ARRAY_SIZES[0]:,}."
        )

    else:

        print(
            "No slower-to-faster crossover "
            "occurred within the tested N range."
        )

    print()
    print(
        "Result verification: PASSED"
    )


if __name__ == "__main__":
    main()
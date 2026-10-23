import os
import sys
import time
import statistics

import numpy as np
from numba import njit, prange, set_num_threads


STUDENT_ID = 230103369
ARRAY_SIZE = 2_000_000
PHYSICAL_CORES = 8
REPETITIONS = 5


@njit
def serial_prefix_scan(values):
    output = np.empty(
        values.size,
        dtype=np.int64,
    )

    running_sum = 0

    for i in range(values.size):
        running_sum += values[i]
        output[i] = running_sum

    return output


@njit(parallel=True)
def local_chunk_scan(
    values,
    output,
    chunk_sums,
    workers,
):
    """
    Pass 1:
    Each worker computes an inclusive prefix scan
    inside its own chunk.
    """

    n = values.size
    chunk_size = (
        n + workers - 1
    ) // workers

    for worker in prange(workers):
        start = worker * chunk_size
        stop = min(
            start + chunk_size,
            n,
        )

        if start < stop:
            running_sum = 0

            for i in range(start, stop):
                running_sum += values[i]
                output[i] = running_sum

            chunk_sums[worker] = running_sum


@njit
def compute_offsets(chunk_sums):
    """
    Offset reduction between chunks.
    """

    offsets = np.zeros(
        chunk_sums.size,
        dtype=np.int64,
    )

    running_sum = 0

    for i in range(chunk_sums.size):
        offsets[i] = running_sum
        running_sum += chunk_sums[i]

    return offsets


@njit(parallel=True)
def apply_offsets(
    output,
    offsets,
    workers,
):
    """
    Pass 2:
    Apply the sum of all previous chunks
    to every element in the current chunk.
    """

    n = output.size
    chunk_size = (
        n + workers - 1
    ) // workers

    for worker in prange(workers):
        start = worker * chunk_size
        stop = min(
            start + chunk_size,
            n,
        )

        offset = offsets[worker]

        if offset != 0:
            for i in range(start, stop):
                output[i] += offset


def parallel_prefix_scan(values, workers):
    output = np.empty(
        values.size,
        dtype=np.int64,
    )

    chunk_sums = np.zeros(
        workers,
        dtype=np.int64,
    )

    # Pass 1: local prefix scans.
    local_chunk_scan(
        values,
        output,
        chunk_sums,
        workers,
    )

    # Reduction: calculate chunk offsets.
    offsets = compute_offsets(
        chunk_sums
    )

    # Pass 2: apply offsets.
    apply_offsets(
        output,
        offsets,
        workers,
    )

    return output


def benchmark_serial(values):
    times = []
    result = None

    for _ in range(REPETITIONS):
        start = time.perf_counter()

        result = serial_prefix_scan(
            values
        )

        end = time.perf_counter()

        times.append(
            (end - start) * 1000.0
        )

    return (
        result,
        statistics.median(times),
    )


def benchmark_parallel(values, workers):
    times = []
    result = None

    for _ in range(REPETITIONS):
        start = time.perf_counter()

        result = parallel_prefix_scan(
            values,
            workers,
        )

        end = time.perf_counter()

        times.append(
            (end - start) * 1000.0
        )

    return (
        result,
        statistics.median(times),
    )


def main():
    logical_cores = os.cpu_count() or 1

    set_num_threads(
        PHYSICAL_CORES
    )

    print(f"Student ID: {STUDENT_ID}")
    print(f"Array size: {ARRAY_SIZE}")
    print("Input values: integers 1 to 10")
    print(
        f"Logical core count: "
        f"{logical_cores}"
    )
    print(
        f"Parallel workers: "
        f"{PHYSICAL_CORES}"
    )
    print(
        f"Timing statistic: median of "
        f"{REPETITIONS} runs"
    )
    print()

    rng = np.random.default_rng(
        STUDENT_ID
    )

    values = rng.integers(
        1,
        11,
        size=ARRAY_SIZE,
        dtype=np.int64,
    )

    # JIT compilation warm-up.
    warmup = values[:100]

    serial_prefix_scan(warmup)

    warmup_output = np.empty(
        warmup.size,
        dtype=np.int64,
    )

    warmup_chunk_sums = np.zeros(
        PHYSICAL_CORES,
        dtype=np.int64,
    )

    local_chunk_scan(
        warmup,
        warmup_output,
        warmup_chunk_sums,
        PHYSICAL_CORES,
    )

    warmup_offsets = compute_offsets(
        warmup_chunk_sums
    )

    apply_offsets(
        warmup_output,
        warmup_offsets,
        PHYSICAL_CORES,
    )

    # Benchmark.
    serial_output, serial_time = (
        benchmark_serial(values)
    )

    parallel_output, parallel_time = (
        benchmark_parallel(
            values,
            PHYSICAL_CORES,
        )
    )

    # Required verification constraint.
    try:
        assert (
            parallel_output[-1]
            == serial_output[-1]
        )

        verification = "PASS"

    except AssertionError:
        verification = "FAIL"

        print("Parallel Prefix Scan")
        print("-" * 65)
        print(
            "Verification Status: FAIL"
        )

        sys.exit(1)

    speedup = (
        serial_time / parallel_time
    )

    print("Parallel Prefix Scan")
    print("-" * 65)

    print(
        "Serial Time (ms) | "
        "Parallel Time (ms) | "
        "Speedup | Verification Status"
    )

    print(
        "-----------------|"
        "--------------------|"
        "---------|--------------------"
    )

    print(
        f"{serial_time:16.3f} | "
        f"{parallel_time:18.3f} | "
        f"{speedup:7.3f}x | "
        f"{verification}"
    )

    print()
    print(
        f"Serial final prefix sum:   "
        f"{serial_output[-1]}"
    )
    print(
        f"Parallel final prefix sum: "
        f"{parallel_output[-1]}"
    )


if __name__ == "__main__":
    main()
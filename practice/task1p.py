import os
import statistics
import time

import numpy as np
from numba import njit, prange, set_num_threads


STUDENT_ID = 230103369
MATRIX_SIZE = 8000
REPETITIONS = 5

# Apple M2 base model: 8 physical CPU cores.
PHYSICAL_CORES = 8


@njit(parallel=True)
def row_major_sum(matrix):
    """
    Mode A:
    Workers split rows.
    Each worker traverses columns linearly.
    """
    partial_sums = np.zeros(
        matrix.shape[0],
        dtype=np.float64,
    )

    for i in prange(matrix.shape[0]):
        local_sum = 0.0

        for j in range(matrix.shape[1]):
            local_sum += matrix[i, j]

        partial_sums[i] = local_sum

    return np.sum(partial_sums)


@njit(parallel=True)
def column_major_sum(matrix):
    """
    Mode B:
    Workers split columns.
    Each worker traverses rows using strided access.
    """
    partial_sums = np.zeros(
        matrix.shape[1],
        dtype=np.float64,
    )

    for j in prange(matrix.shape[1]):
        local_sum = 0.0

        for i in range(matrix.shape[0]):
            local_sum += matrix[i, j]

        partial_sums[j] = local_sum

    return np.sum(partial_sums)


def benchmark(function, matrix):
    """
    Run the kernel multiple times and return
    the median wall-clock time.
    """
    times = []
    result = 0.0

    for _ in range(REPETITIONS):
        start = time.perf_counter()

        result = function(matrix)

        end = time.perf_counter()

        times.append(
            (end - start) * 1000.0
        )

    return result, statistics.median(times)


def main():
    logical_cores = os.cpu_count() or 1

    set_num_threads(PHYSICAL_CORES)

    print(f"Student ID: {STUDENT_ID}")
    print(
        f"Matrix size: "
        f"{MATRIX_SIZE} x {MATRIX_SIZE}"
    )
    print("Data type: float64")
    print(
        f"Logical core count: "
        f"{logical_cores}"
    )
    print(
        f"Physical cores / worker threads: "
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

    matrix = rng.random(
        (MATRIX_SIZE, MATRIX_SIZE),
        dtype=np.float64,
    )

    # Compile both Numba kernels before timing.
    warmup = matrix[:10, :10]

    row_major_sum(warmup)
    column_major_sum(warmup)

    row_sum, row_time = benchmark(
        row_major_sum,
        matrix,
    )

    column_sum, column_time = benchmark(
        column_major_sum,
        matrix,
    )

    if not np.isclose(
        row_sum,
        column_sum,
        rtol=1e-10,
        atol=1e-6,
    ):
        raise RuntimeError(
            "Sum verification failed."
        )

    cache_penalty = (
        column_time / row_time
    )

    print("Parallel Cache Traversal")
    print("-" * 55)

    print(
        f"Mode A - Row-Major:            "
        f"{row_time:.3f} ms"
    )

    print(
        f"Mode B - Column-Major/Strided: "
        f"{column_time:.3f} ms"
    )

    print(
        f"Cache penalty factor:          "
        f"{cache_penalty:.3f}x"
    )

    print()
    print(
        f"Mode A sum: "
        f"{row_sum:.10f}"
    )
    print(
        f"Mode B sum: "
        f"{column_sum:.10f}"
    )
    print(
        "Result verification: PASSED"
    )


if __name__ == "__main__":
    main()
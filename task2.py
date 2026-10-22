import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np


STUDENT_ID = 230103369
ARRAY_SIZE = 20_000_000
PARALLEL_FRACTION = 0.95


def partial_sum(chunk):
    total = 0.0
    for value in chunk:
        total += value
    return total


def parallel_sum(data, workers):
    chunks = np.array_split(data, workers)

    start_time = time.perf_counter()

    if workers == 1:
        result = partial_sum(chunks[0])
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            partial_results = executor.map(partial_sum, chunks)
            result = sum(partial_results)

    end_time = time.perf_counter()

    elapsed_ms = (end_time - start_time) * 1000

    return result, elapsed_ms


def amdahl_speedup(workers):
    return 1 / ((1 - PARALLEL_FRACTION) +
                (PARALLEL_FRACTION / workers))


def main():
    max_cores = os.cpu_count() or 1
    worker_counts = [1, 2, 4, max_cores, max_cores * 2]


    worker_counts = list(dict.fromkeys(worker_counts))

    rng = np.random.default_rng(STUDENT_ID)
    data = rng.random(ARRAY_SIZE, dtype=np.float64)

    print(f"Student ID: {STUDENT_ID}")
    print(f"Logical cores: {max_cores}")
    print(f"Array size: {ARRAY_SIZE}")
    print(f"Data type: {data.dtype}")
    print(f"Seed: {STUDENT_ID}")
    print(f"Parallel fraction P: {PARALLEL_FRACTION}")
    print()

    results = []

    baseline_time = None

    for workers in worker_counts:
        total, elapsed_ms = parallel_sum(data, workers)

        if workers == 1:
            baseline_time = elapsed_ms

        speedup = baseline_time / elapsed_ms
        amdahl = amdahl_speedup(workers)

        results.append(
            (workers, elapsed_ms, speedup, amdahl, total)
        )

    print(
        f"{'Cores':<8}"
        f"{'Wall Time (ms)':<18}"
        f"{'Speedup':<12}"
        f"{'Amdahl Upper Bound':<20}"
    )

    print("-" * 58)

    for workers, elapsed_ms, speedup, amdahl, total in results:
        print(
            f"{workers:<8}"
            f"{elapsed_ms:<18.3f}"
            f"{speedup:<12.3f}"
            f"{amdahl:<20.3f}"
        )

    print()
    print(f"Reference sum: {results[0][4]:.10f}")


if __name__ == "__main__":
    main()
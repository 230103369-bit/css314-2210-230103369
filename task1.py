import os
import platform
import time

import numpy as np


STUDENT_ID = 230103369
ARRAY_SIZE = 20_000_000


def main():

    print(f"Student ID: {STUDENT_ID}")
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"Platform: {platform.platform()}")
    print(f"CPU/Processor: {platform.processor() or platform.machine()}")
    print(f"Logical cores: {os.cpu_count()}")

    rng = np.random.default_rng(STUDENT_ID)
    data = rng.random(ARRAY_SIZE, dtype=np.float64)

    print(f"Array size: {data.size}")
    print(f"Data type: {data.dtype}")
    print(f"Seed: {STUDENT_ID}")


    start_time = time.perf_counter()

    total = 0.0
    for value in data:
        total += value

    mean = total / ARRAY_SIZE

    squared_diff_sum = 0.0
    for value in data:
        diff = value - mean
        squared_diff_sum += diff * diff

    std_dev = (squared_diff_sum / ARRAY_SIZE) ** 0.5

    end_time = time.perf_counter()

    elapsed_ms = (end_time - start_time) * 1000

    # Results
    print(f"Sum: {total:.10f}")
    print(f"Standard deviation: {std_dev:.10f}")
    print(f"Execution time: {elapsed_ms:.3f} ms")


if __name__ == "__main__":
    main()
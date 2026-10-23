import multiprocessing as mp
from multiprocessing import shared_memory
import statistics
import time

import numpy as np


STUDENT_ID = 230103369

SIZES = [
    100_000,
    1_000_000,
    10_000_000,
]

REPETITIONS = 5
DTYPE = np.float64


def pipe_worker(connection):
    """
    Path A worker.

    Receives a NumPy array through multiprocessing Pipe.
    The array is serialized and transferred through IPC.
    """

    while True:
        message = connection.recv()

        if message is None:
            break

        array = message

        # Read the transferred data so that the worker
        # actually consumes the received array.
        checksum = float(array[0] + array[-1])

        connection.send(checksum)

    connection.close()


def shared_memory_worker(connection):
    """
    Path B worker.

    Receives only shared-memory metadata.
    The NumPy array itself is not serialized or copied
    between the parent and worker.
    """

    while True:
        message = connection.recv()

        if message is None:
            break

        shm_name, size, dtype_string = message

        shm = shared_memory.SharedMemory(
            name=shm_name
        )

        try:
            array = np.ndarray(
                (size,),
                dtype=np.dtype(dtype_string),
                buffer=shm.buf,
            )

            # Access the shared data.
            checksum = float(
                array[0] + array[-1]
            )

            connection.send(checksum)

        finally:
            shm.close()

    connection.close()


def benchmark_pipe(
    connection,
    array,
):
    """
    Measure Path A:
    serialize + IPC transfer + acknowledgement.
    """

    times = []

    for _ in range(REPETITIONS):
        start = time.perf_counter()

        connection.send(array)
        checksum = connection.recv()

        elapsed_ms = (
            time.perf_counter() - start
        ) * 1000.0

        expected = float(
            array[0] + array[-1]
        )

        if checksum != expected:
            raise RuntimeError(
                "Pipe verification failed"
            )

        times.append(elapsed_ms)

    return statistics.median(times)


def benchmark_shared_memory(
    connection,
    array,
):
    """
    Measure Path B:
    place data in shared memory once, then measure
    zero-copy IPC access using shared-memory metadata.
    """

    shm = shared_memory.SharedMemory(
        create=True,
        size=array.nbytes,
    )

    shared_array = np.ndarray(
        array.shape,
        dtype=array.dtype,
        buffer=shm.buf,
    )

    # Populate the shared region before timing IPC access.
    np.copyto(
        shared_array,
        array,
    )

    times = []

    try:
        message = (
            shm.name,
            array.size,
            array.dtype.str,
        )

        for _ in range(REPETITIONS):
            start = time.perf_counter()

            connection.send(message)
            checksum = connection.recv()

            elapsed_ms = (
                time.perf_counter() - start
            ) * 1000.0

            expected = float(
                array[0] + array[-1]
            )

            if checksum != expected:
                raise RuntimeError(
                    "Shared memory verification failed"
                )

            times.append(elapsed_ms)

    finally:
        shm.close()
        shm.unlink()

    return statistics.median(times)


def main():
    # "spawn" is the appropriate multiprocessing
    # start method for macOS.
    context = mp.get_context("spawn")

    pipe_parent, pipe_child = (
        context.Pipe()
    )

    shm_parent, shm_child = (
        context.Pipe()
    )

    pipe_process = context.Process(
        target=pipe_worker,
        args=(pipe_child,),
    )

    shm_process = context.Process(
        target=shared_memory_worker,
        args=(shm_child,),
    )

    pipe_process.start()
    shm_process.start()

    # Parent does not use child endpoints.
    pipe_child.close()
    shm_child.close()

    try:
        # Warm up both already-running workers.
        warmup_array = np.array(
            [1.0, 2.0],
            dtype=DTYPE,
        )

        pipe_parent.send(warmup_array)
        pipe_parent.recv()

        warmup_shm = (
            shared_memory.SharedMemory(
                create=True,
                size=warmup_array.nbytes,
            )
        )

        try:
            warmup_view = np.ndarray(
                warmup_array.shape,
                dtype=warmup_array.dtype,
                buffer=warmup_shm.buf,
            )

            np.copyto(
                warmup_view,
                warmup_array,
            )

            shm_parent.send(
                (
                    warmup_shm.name,
                    warmup_array.size,
                    warmup_array.dtype.str,
                )
            )

            shm_parent.recv()

        finally:
            warmup_shm.close()
            warmup_shm.unlink()

        rng = np.random.default_rng(
            STUDENT_ID
        )

        print(
            f"Student ID: {STUDENT_ID}"
        )
        print(
            f"Data type: {np.dtype(DTYPE)}"
        )
        print(
            f"Timing statistic: median of "
            f"{REPETITIONS} runs"
        )
        print()
        print("IPC Memory Transfer Profile")
        print("-" * 72)
        print(
            "N          | Array Size (MB) | "
            "Path A Pipe (ms) | "
            "Path B Shared Memory (ms) | "
            "A/B Ratio"
        )
        print(
            "-----------|-----------------|"
            "------------------|"
            "---------------------------|"
            "----------"
        )

        for size in SIZES:
            array = rng.random(
                size,
                dtype=DTYPE,
            )

            pipe_time = benchmark_pipe(
                pipe_parent,
                array,
            )

            shared_time = (
                benchmark_shared_memory(
                    shm_parent,
                    array,
                )
            )

            array_mb = (
                array.nbytes /
                (1024.0 * 1024.0)
            )

            ratio = (
                pipe_time /
                shared_time
            )

            print(
                f"{size:10d} | "
                f"{array_mb:15.3f} | "
                f"{pipe_time:16.3f} | "
                f"{shared_time:25.3f} | "
                f"{ratio:8.3f}x"
            )

        print()
        print(
            "Path A: Pipe IPC serialization / copy"
        )
        print(
            "Path B: Zero-copy shared-memory access"
        )
        print(
            "Result verification: PASSED"
        )

    finally:
        # Gracefully stop workers.
        try:
            pipe_parent.send(None)
        except Exception:
            pass

        try:
            shm_parent.send(None)
        except Exception:
            pass

        pipe_process.join()
        shm_process.join()

        pipe_parent.close()
        shm_parent.close()


if __name__ == "__main__":
    main()
import ctypes
import os
import platform
import subprocess
import tempfile
import time


STUDENT_ID = 230103369
INCREMENTS = 50_000_000

# Benchmark machine: MacBook Air with Apple M2.
# Apple M2 has an 8-core CPU (4 performance + 4 efficiency cores).
# Task 3 explicitly requires N to equal the physical core count,
# so the worker count is fixed to 8.
WORKERS = 8


# Native C helper used to demonstrate real cache-line contention.
#
# Python threads are not suitable for this CPU-bound benchmark because
# of the GIL. The helper uses native POSIX threads so multiple workers
# can execute concurrently on different CPU cores.
#
# volatile is used so the compiler cannot optimize the repeated memory
# updates into a single final assignment.
C_SOURCE = r"""
#include <pthread.h>
#include <stdint.h>
#include <stdlib.h>


typedef struct {
    volatile uint64_t *shared_counts;
    int index;
    uint64_t iterations;
    int local_mode;
} thread_arg_t;


static void *worker_function(void *arg_ptr)
{
    thread_arg_t *arg = (thread_arg_t *)arg_ptr;

    if (arg->local_mode) {
        /*
         * Local version:
         * increment a thread-local variable and write to the shared
         * array only once after the loop has finished.
         */
        volatile uint64_t local_count = 0;

        for (uint64_t i = 0; i < arg->iterations; ++i) {
            local_count++;
        }

        arg->shared_counts[arg->index] = local_count;
    }
    else {
        /*
         * Unpadded shared version:
         * neighboring uint64_t elements are only 8 bytes apart.
         * Multiple counters can therefore occupy the same cache line.
         *
         * Each increment writes directly to shared memory.
         */
        volatile uint64_t *counter =
            &arg->shared_counts[arg->index];

        for (uint64_t i = 0; i < arg->iterations; ++i) {
            (*counter)++;
        }
    }

    return NULL;
}


int run_benchmark(
    int workers,
    uint64_t iterations,
    int local_mode,
    uint64_t *output
)
{
    pthread_t *threads =
        (pthread_t *)malloc(sizeof(pthread_t) * workers);

    thread_arg_t *args =
        (thread_arg_t *)malloc(sizeof(thread_arg_t) * workers);

    if (threads == NULL || args == NULL) {
        free(threads);
        free(args);
        return -1;
    }

    for (int i = 0; i < workers; ++i) {
        output[i] = 0;

        args[i].shared_counts = output;
        args[i].index = i;
        args[i].iterations = iterations;
        args[i].local_mode = local_mode;

        if (pthread_create(
                &threads[i],
                NULL,
                worker_function,
                &args[i]
            ) != 0) {

            free(threads);
            free(args);
            return -2;
        }
    }

    for (int i = 0; i < workers; ++i) {
        pthread_join(threads[i], NULL);
    }

    free(threads);
    free(args);

    return 0;
}
"""


def build_native_helper(temp_dir):
    source_path = os.path.join(
        temp_dir,
        "false_sharing.c"
    )

    library_path = os.path.join(
        temp_dir,
        "false_sharing.dylib"
    )

    with open(source_path, "w", encoding="utf-8") as file:
        file.write(C_SOURCE)

    command = [
        "clang",
        "-O2",
        "-dynamiclib",
        "-pthread",
        source_path,
        "-o",
        library_path,
    ]

    subprocess.run(
        command,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    return library_path


def load_native_helper(library_path):
    library = ctypes.CDLL(library_path)

    library.run_benchmark.argtypes = [
        ctypes.c_int,
        ctypes.c_uint64,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_uint64),
    ]

    library.run_benchmark.restype = ctypes.c_int

    return library


def run_benchmark(library, iterations, local_mode):
    result_type = ctypes.c_uint64 * WORKERS
    results = result_type()

    start = time.perf_counter()

    status = library.run_benchmark(
        WORKERS,
        iterations,
        local_mode,
        results,
    )

    end = time.perf_counter()

    if status != 0:
        raise RuntimeError(
            f"Native benchmark failed with status {status}"
        )

    elapsed_ms = (end - start) * 1000

    return list(results), elapsed_ms


def main():
    print(f"Student ID: {STUDENT_ID}")
    print("CPU: Apple M2")
    print(f"Architecture: {platform.machine()}")
    print(f"Physical cores / workers: {WORKERS}")
    print(f"Increments per worker: {INCREMENTS}")
    print()

    with tempfile.TemporaryDirectory() as temp_dir:
        library_path = build_native_helper(temp_dir)
        library = load_native_helper(library_path)

        # Warm-up runs are intentionally excluded from the benchmark.
        # They remove one-time thread/library startup effects.
        run_benchmark(
            library,
            10_000,
            local_mode=0,
        )

        run_benchmark(
            library,
            10_000,
            local_mode=1,
        )

        # Benchmark 1:
        # every worker repeatedly updates its neighboring shared counter.
        unpadded_result, unpadded_time = run_benchmark(
            library,
            INCREMENTS,
            local_mode=0,
        )

        # Benchmark 2:
        # every worker increments a local variable and performs only
        # one shared-memory write at the end.
        local_result, local_time = run_benchmark(
            library,
            INCREMENTS,
            local_mode=1,
        )

    expected = INCREMENTS

    unpadded_correct = all(
        value == expected
        for value in unpadded_result
    )

    local_correct = all(
        value == expected
        for value in local_result
    )

    ratio = unpadded_time / local_time

    print("False Sharing Benchmark")
    print("-" * 45)

    print(
        f"Unpadded Shared: {unpadded_time:.3f} ms"
    )

    print(
        f"Padded/Local:    {local_time:.3f} ms"
    )

    print(
        f"Execution ratio: {ratio:.3f}x"
    )

    print()

    print(
        f"Expected count per worker: {expected}"
    )

    print(
        f"Unpadded result correct: {unpadded_correct}"
    )

    print(
        f"Local result correct:    {local_correct}"
    )


if __name__ == "__main__":
    main()
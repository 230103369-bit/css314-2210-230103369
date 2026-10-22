import ctypes
import os
import subprocess
import tempfile
import time


STUDENT_ID = 230103369

THREADS = 4
INCREMENTS_PER_THREAD = 2_000_000
EXPECTED_VALUE = THREADS * INCREMENTS_PER_THREAD

REPEATS = 5


C_SOURCE = r"""
#include <pthread.h>
#include <stdint.h>
#include <stdlib.h>


typedef struct {
    volatile uint64_t *counter;
    pthread_mutex_t *mutex;
    uint64_t increments;
    int synchronized;
} thread_arg_t;


static void *counter_worker(void *arg_ptr)
{
    thread_arg_t *arg = (thread_arg_t *)arg_ptr;

    if (arg->synchronized) {

        for (uint64_t i = 0; i < arg->increments; ++i) {

            pthread_mutex_lock(arg->mutex);

            (*arg->counter)++;

            pthread_mutex_unlock(arg->mutex);
        }

    } else {

        for (uint64_t i = 0; i < arg->increments; ++i) {

            /*
             * Intentionally unsafe read-modify-write.
             * Multiple threads can overwrite each other's updates.
             */
            uint64_t value = *arg->counter;
            value++;
            *arg->counter = value;
        }
    }

    return NULL;
}


uint64_t run_counter_test(
    int thread_count,
    uint64_t increments,
    int synchronized
)
{
    volatile uint64_t counter = 0;

    pthread_t *threads =
        (pthread_t *)malloc(
            sizeof(pthread_t) * thread_count
        );

    thread_arg_t *args =
        (thread_arg_t *)malloc(
            sizeof(thread_arg_t) * thread_count
        );

    if (threads == NULL || args == NULL) {
        free(threads);
        free(args);
        return 0;
    }

    pthread_mutex_t mutex;
    pthread_mutex_init(&mutex, NULL);

    for (int i = 0; i < thread_count; ++i) {

        args[i].counter = &counter;
        args[i].mutex = &mutex;
        args[i].increments = increments;
        args[i].synchronized = synchronized;

        pthread_create(
            &threads[i],
            NULL,
            counter_worker,
            &args[i]
        );
    }

    for (int i = 0; i < thread_count; ++i) {
        pthread_join(
            threads[i],
            NULL
        );
    }

    pthread_mutex_destroy(&mutex);

    free(threads);
    free(args);

    return counter;
}
"""


def build_native_helper(temp_dir):
    source_path = os.path.join(
        temp_dir,
        "counter_benchmark.c",
    )

    library_path = os.path.join(
        temp_dir,
        "counter_benchmark.dylib",
    )

    with open(
        source_path,
        "w",
        encoding="utf-8",
    ) as file:
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


def load_library(library_path):
    library = ctypes.CDLL(
        library_path
    )

    library.run_counter_test.argtypes = [
        ctypes.c_int,
        ctypes.c_uint64,
        ctypes.c_int,
    ]

    library.run_counter_test.restype = (
        ctypes.c_uint64
    )

    return library


def run_once(library, synchronized):
    start = time.perf_counter()

    result = library.run_counter_test(
        THREADS,
        INCREMENTS_PER_THREAD,
        synchronized,
    )

    end = time.perf_counter()

    elapsed_ms = (
        end - start
    ) * 1000

    return result, elapsed_ms


def main():
    print(
        f"Student ID: {STUDENT_ID}"
    )

    print(
        f"Threads: {THREADS}"
    )

    print(
        "Increments per thread: "
        f"{INCREMENTS_PER_THREAD}"
    )

    print(
        f"Expected final value: "
        f"{EXPECTED_VALUE}"
    )

    print()

    unsafe_results = []
    unsafe_times = []

    # Run several real unsafe trials because race-condition
    # results can vary between executions.
    for trial in range(
        1,
        REPEATS + 1,
    ):
        actual_value, elapsed_ms = run_once(
            library,
            0,
        )

        lost_updates = (
            EXPECTED_VALUE - actual_value
        )

        lost_percentage = (
            lost_updates
            / EXPECTED_VALUE
        ) * 100

        unsafe_results.append(
            actual_value
        )

        unsafe_times.append(
            elapsed_ms
        )

        print(
            f"Unsafe trial {trial}: "
            f"actual={actual_value}, "
            f"lost={lost_percentage:.3f}%, "
            f"time={elapsed_ms:.3f} ms"
        )

    print()

    # Use the final unsafe trial as the reported
    # actual value for the required summary.
    unsafe_actual = unsafe_results[-1]

    lost_updates = (
        EXPECTED_VALUE - unsafe_actual
    )

    lost_percentage = (
        lost_updates
        / EXPECTED_VALUE
    ) * 100

    print("Race Condition Summary")
    print("-" * 45)

    print(
        f"Expected final value: "
        f"{EXPECTED_VALUE}"
    )

    print(
        f"Actual final value:   "
        f"{unsafe_actual}"
    )

    print(
        f"Updates lost:         "
        f"{lost_updates}"
    )

    print(
        f"Percentage lost:      "
        f"{lost_percentage:.3f}%"
    )

    print()

    # Correct synchronized version.
    locked_actual, locked_time = run_once(
        library,
        1,
    )

    # Compare with the final unsafe trial.
    unsafe_time = unsafe_times[-1]

    time_penalty_ms = (
        locked_time - unsafe_time
    )

    if unsafe_time > 0:
        time_penalty_ratio = (
            locked_time / unsafe_time
        )
    else:
        time_penalty_ratio = float(
            "inf"
        )

    print("Synchronized Version")
    print("-" * 45)

    print(
        f"Final value with lock: "
        f"{locked_actual}"
    )

    print(
        f"Correct result: "
        f"{locked_actual == EXPECTED_VALUE}"
    )

    print(
        f"Unsafe wall time: "
        f"{unsafe_time:.3f} ms"
    )

    print(
        f"Locked wall time: "
        f"{locked_time:.3f} ms"
    )

    print(
        f"Synchronization penalty: "
        f"{time_penalty_ms:.3f} ms"
    )

    print(
        f"Synchronization slowdown: "
        f"{time_penalty_ratio:.3f}x"
    )


if __name__ == "__main__":

    with tempfile.TemporaryDirectory() as temp_dir:

        library_path = build_native_helper(
            temp_dir
        )

        library = load_library(
            library_path
        )

        main()
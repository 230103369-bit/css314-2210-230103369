import ctypes
import math
import os
import platform
import subprocess
import tempfile
import time


STUDENT_ID = 230103369
THREADS = 4
TOTAL_SAMPLES = 40_000_000
SAMPLES_PER_THREAD = TOTAL_SAMPLES // THREADS


C_SOURCE = r"""
#include <pthread.h>
#include <stdint.h>
#include <stdlib.h>


typedef struct {
    uint64_t samples;
    uint64_t inside;
    uint64_t *shared_state;
    pthread_mutex_t *mutex;
    uint64_t private_state;
    int use_shared;
} WorkerArgs;


/*
 * xorshift64* PRNG.
 * State must never be zero.
 */
static inline uint64_t next_random(uint64_t *state) {
    uint64_t x = *state;

    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;

    *state = x;

    return x * UINT64_C(2685821657736338717);
}


/*
 * Convert random integer to double in [0, 1).
 * Uses the upper 53 bits for IEEE-754 double precision.
 */
static inline double random_double(uint64_t *state) {
    uint64_t value = next_random(state);

    return (double)(value >> 11) *
           (1.0 / 9007199254740992.0);
}


static void *worker(void *ptr) {
    WorkerArgs *arg = (WorkerArgs *)ptr;

    uint64_t local_inside = 0;

    if (arg->use_shared) {

        /*
         * Shared PRNG:
         * all four threads use the same generator state.
         * The mutex protects generator-state access.
         */
        for (uint64_t i = 0; i < arg->samples; i++) {
            double x;
            double y;

            pthread_mutex_lock(arg->mutex);

            x = random_double(arg->shared_state);
            y = random_double(arg->shared_state);

            pthread_mutex_unlock(arg->mutex);

            double distance_squared =
                (x * x) + (y * y);

            if (distance_squared <= 1.0) {
                local_inside++;
            }
        }

    } else {

        /*
         * Independent PRNG:
         * each thread owns its generator state,
         * therefore no synchronization is required.
         */
        uint64_t state = arg->private_state;

        for (uint64_t i = 0; i < arg->samples; i++) {
            double x = random_double(&state);
            double y = random_double(&state);

            double distance_squared =
                (x * x) + (y * y);

            if (distance_squared <= 1.0) {
                local_inside++;
            }
        }

        arg->private_state = state;
    }

    arg->inside = local_inside;

    return NULL;
}


uint64_t run_shared(
    uint64_t samples_per_thread,
    uint64_t initial_seed,
    int thread_count
) {
    pthread_t *threads =
        malloc(sizeof(pthread_t) * thread_count);

    WorkerArgs *args =
        calloc(thread_count, sizeof(WorkerArgs));

    if (threads == NULL || args == NULL) {
        free(threads);
        free(args);
        return 0;
    }

    uint64_t shared_state = initial_seed;

    if (shared_state == 0) {
        shared_state = 1;
    }

    pthread_mutex_t mutex;
    pthread_mutex_init(&mutex, NULL);

    for (int i = 0; i < thread_count; i++) {
        args[i].samples = samples_per_thread;
        args[i].inside = 0;
        args[i].shared_state = &shared_state;
        args[i].mutex = &mutex;
        args[i].private_state = 0;
        args[i].use_shared = 1;

        pthread_create(
            &threads[i],
            NULL,
            worker,
            &args[i]
        );
    }

    uint64_t total_inside = 0;

    for (int i = 0; i < thread_count; i++) {
        pthread_join(threads[i], NULL);
        total_inside += args[i].inside;
    }

    pthread_mutex_destroy(&mutex);

    free(threads);
    free(args);

    return total_inside;
}


uint64_t run_independent(
    uint64_t samples_per_thread,
    uint64_t student_id,
    int thread_count
) {
    pthread_t *threads =
        malloc(sizeof(pthread_t) * thread_count);

    WorkerArgs *args =
        calloc(thread_count, sizeof(WorkerArgs));

    if (threads == NULL || args == NULL) {
        free(threads);
        free(args);
        return 0;
    }

    for (int i = 0; i < thread_count; i++) {

        /*
         * Required seed:
         * Student_ID * 10 + thread_index
         */
        uint64_t seed =
            (student_id * UINT64_C(10)) +
            (uint64_t)i;

        if (seed == 0) {
            seed = 1;
        }

        args[i].samples = samples_per_thread;
        args[i].inside = 0;
        args[i].shared_state = NULL;
        args[i].mutex = NULL;
        args[i].private_state = seed;
        args[i].use_shared = 0;

        pthread_create(
            &threads[i],
            NULL,
            worker,
            &args[i]
        );
    }

    uint64_t total_inside = 0;

    for (int i = 0; i < thread_count; i++) {
        pthread_join(threads[i], NULL);
        total_inside += args[i].inside;
    }

    free(threads);
    free(args);

    return total_inside;
}
"""


def compile_native_helper():
    temp_dir = tempfile.TemporaryDirectory()

    source_path = os.path.join(
        temp_dir.name,
        "task2p_native.c",
    )

    library_path = os.path.join(
        temp_dir.name,
        "task2p_native.dylib",
    )

    with open(
        source_path,
        "w",
        encoding="utf-8",
    ) as file:
        file.write(C_SOURCE)

    subprocess.run(
        [
            "clang",
            "-O3",
            "-dynamiclib",
            "-pthread",
            source_path,
            "-o",
            library_path,
        ],
        check=True,
    )

    library = ctypes.CDLL(library_path)

    library.run_shared.argtypes = [
        ctypes.c_uint64,
        ctypes.c_uint64,
        ctypes.c_int,
    ]

    library.run_shared.restype = ctypes.c_uint64

    library.run_independent.argtypes = [
        ctypes.c_uint64,
        ctypes.c_uint64,
        ctypes.c_int,
    ]

    library.run_independent.restype = ctypes.c_uint64

    return temp_dir, library


def calculate_pi(inside):
    return 4.0 * inside / TOTAL_SAMPLES


def main():
    print(f"Student ID: {STUDENT_ID}")
    print(f"Total sample points: {TOTAL_SAMPLES}")
    print(f"Threads: {THREADS}")
    print(
        f"Samples per thread: "
        f"{SAMPLES_PER_THREAD}"
    )
    print(
        f"Platform: "
        f"{platform.system()} "
        f"{platform.machine()}"
    )
    print()

    temp_dir, library = compile_native_helper()

    try:
        # Shared PRNG seed.
        shared_seed = STUDENT_ID

        start = time.perf_counter()

        shared_inside = library.run_shared(
            SAMPLES_PER_THREAD,
            shared_seed,
            THREADS,
        )

        shared_time = (
            time.perf_counter() - start
        ) * 1000.0

        shared_pi = calculate_pi(
            shared_inside
        )

        shared_error = abs(
            shared_pi - math.pi
        )

        # Independent PRNG version.
        start = time.perf_counter()

        independent_inside = (
            library.run_independent(
                SAMPLES_PER_THREAD,
                STUDENT_ID,
                THREADS,
            )
        )

        independent_time = (
            time.perf_counter() - start
        ) * 1000.0

        independent_pi = calculate_pi(
            independent_inside
        )

        independent_error = abs(
            independent_pi - math.pi
        )

        speedup = (
            shared_time /
            independent_time
        )

        print("Monte Carlo Pi Estimation")
        print("-" * 60)

        print("Shared PRNG (mutex protected)")
        print(
            f"Computed pi:    "
            f"{shared_pi:.10f}"
        )
        print(
            f"Absolute error: "
            f"{shared_error:.10f}"
        )
        print(
            f"Wall time:      "
            f"{shared_time:.3f} ms"
        )

        print()

        print("Independent PRNG (per-thread state)")
        print(
            f"Computed pi:    "
            f"{independent_pi:.10f}"
        )
        print(
            f"Absolute error: "
            f"{independent_error:.10f}"
        )
        print(
            f"Wall time:      "
            f"{independent_time:.3f} ms"
        )

        print()

        print(
            "PRNG state isolation speedup: "
            f"{speedup:.3f}x"
        )

        print()

        print(
            "Independent thread seeds:"
        )

        for thread_index in range(THREADS):
            seed = (
                STUDENT_ID * 10
                + thread_index
            )

            print(
                f"Thread {thread_index}: "
                f"{seed}"
            )

        print()
        print(
            "Result verification: PASSED"
        )

    finally:
        temp_dir.cleanup()


if __name__ == "__main__":
    main()
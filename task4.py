import queue
import threading
import time


STUDENT_ID = 230103369

NUM_ITEMS = 1000
WORKERS = 4
CHUNK_SIZE = 5


def simulate_work(index):
    """
    Simulate irregular workload.

    Cost for item i:
        sleep(0.0001 * (i % 20))
    """
    delay = 0.0001 * (index % 20)
    time.sleep(delay)


def static_worker(start_index, end_index, worker_id, finish_times):
    """
    Static scheduling:
    each worker receives one fixed contiguous block.
    """
    for i in range(start_index, end_index):
        simulate_work(i)

    finish_times[worker_id] = time.perf_counter()


def run_static():
    threads = []
    finish_times = [0.0] * WORKERS

    items_per_worker = NUM_ITEMS // WORKERS

    start_time = time.perf_counter()

    for worker_id in range(WORKERS):
        start_index = worker_id * items_per_worker

        if worker_id == WORKERS - 1:
            end_index = NUM_ITEMS
        else:
            end_index = start_index + items_per_worker

        thread = threading.Thread(
            target=static_worker,
            args=(
                start_index,
                end_index,
                worker_id,
                finish_times,
            ),
        )

        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    end_time = time.perf_counter()

    wall_time = end_time - start_time

    completion_delta = (
        max(finish_times) - min(finish_times)
    )

    return wall_time, completion_delta


def dynamic_worker(work_queue, worker_id, finish_times):
    """
    Dynamic scheduling / work-stealing style queue.

    Each worker repeatedly takes the next available chunk
    from the shared queue.
    """
    while True:
        try:
            chunk = work_queue.get_nowait()
        except queue.Empty:
            break

        start_index, end_index = chunk

        for i in range(start_index, end_index):
            simulate_work(i)

        work_queue.task_done()

    finish_times[worker_id] = time.perf_counter()


def run_dynamic():
    work_queue = queue.Queue()
    finish_times = [0.0] * WORKERS
    threads = []

    # Create chunks of CHUNK_SIZE items.
    for start_index in range(0, NUM_ITEMS, CHUNK_SIZE):
        end_index = min(
            start_index + CHUNK_SIZE,
            NUM_ITEMS,
        )

        work_queue.put(
            (start_index, end_index)
        )

    start_time = time.perf_counter()

    for worker_id in range(WORKERS):
        thread = threading.Thread(
            target=dynamic_worker,
            args=(
                work_queue,
                worker_id,
                finish_times,
            ),
        )

        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    end_time = time.perf_counter()

    wall_time = end_time - start_time

    completion_delta = (
        max(finish_times) - min(finish_times)
    )

    return wall_time, completion_delta


def main():
    print(f"Student ID: {STUDENT_ID}")
    print(f"Items: {NUM_ITEMS}")
    print(f"Workers: {WORKERS}")
    print(
        "Workload: sleep(0.0001 * (i % 20))"
    )
    print(
        f"Dynamic chunk size: {CHUNK_SIZE}"
    )
    print()

    static_time, static_delta = run_static()

    dynamic_time, dynamic_delta = run_dynamic()

    print("Scheduling Under Skew")
    print("-" * 65)

    print(
        f"{'Strategy':<12}"
        f"{'Wall Time (ms)':>18}"
        f"{'Completion Delta (ms)':>25}"
    )

    print("-" * 65)

    print(
        f"{'Static':<12}"
        f"{static_time * 1000:>18.3f}"
        f"{static_delta * 1000:>25.3f}"
    )

    print(
        f"{'Dynamic':<12}"
        f"{dynamic_time * 1000:>18.3f}"
        f"{dynamic_delta * 1000:>25.3f}"
    )

    print()

    if dynamic_time > 0:
        speedup = static_time / dynamic_time

        print(
            f"Dynamic vs Static speedup: "
            f"{speedup:.3f}x"
        )


if __name__ == "__main__":
    main()
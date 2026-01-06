"""
Demonstrates two inter-process communication patterns using Python's multiprocessing:
1. Blocking using Events (simulating OS context switches).
2. Spinning using shared memory flags (busy-waiting).

Each pattern runs a "ping-pong" style workload for a defined number of steps, measuring
performance and allowing profiling with tools like py-spy.

Original inspiration: pufferlib
"""

import multiprocessing
import time
import ctypes
import os

# Number of "steps" (ping-pongs) to perform
STEPS = 1_000_000

# ==========================================
# 1. BLOCKING (Anti-Pattern for RL, Good for UI)
# ==========================================
def blocking_worker(start_event, done_event, counter):
    """
    Waits for a signal from the OS.
    In py-spy, you will see time spent in 'futex_wait' or 'acquire'.
    """
    while True:
        start_event.wait()
        start_event.clear()

        # Check termination condition
        if counter.value == -1:
            break

        # Simulate tiny work
        counter.value += 1

        done_event.set()

def run_blocking_experiment():
    print(f"[Blocking Mode] Starting {STEPS} steps...")
    print("  -> Mechanism: multiprocessing.Event (OS Context Switches)")

    start_event = multiprocessing.Event()
    done_event = multiprocessing.Event()
    counter = multiprocessing.Value('i', 0)

    p = multiprocessing.Process(target=blocking_worker, args=(start_event, done_event, counter))
    p.start()

    start_time = time.time()

    for _ in range(STEPS):
        start_event.set()
        done_event.wait()
        done_event.clear()

    end_time = time.time()

    # Cleanup
    counter.value = -1
    start_event.set()
    p.join()

    duration = end_time - start_time
    sps = STEPS / duration
    print(f"[Blocking Mode] Finished in {duration:.4f}s. Speed: {sps:.0f} steps/sec")


# ==========================================
# 2. SPINNING
# ==========================================
def spinning_worker(flag, counter):
    """
    Constantly checks memory.
    In py-spy, you will see time spent IN this function (burning CPU).
    """
    while True:
        # BUSY WAIT: The CPU runs hot here checking the value
        # optimization: No OS sleep, immediate reaction.
        while flag.value == 0:
            pass

        # Check termination
        if flag.value == 2:
            break

        # Simulate tiny work
        counter.value += 1

        # Signal completion by resetting flag
        flag.value = 0

def run_spinning_experiment():
    print(f"[Spinning Mode] Starting {STEPS} steps...")
    print("  -> Mechanism: Shared Memory Flag (Busy Wait)")

    # 0 = Waiting, 1 = Ready, 2 = Stop
    flag = multiprocessing.Value('i', 0)
    counter = multiprocessing.Value('i', 0)

    p = multiprocessing.Process(target=spinning_worker, args=(flag, counter))
    p.start()

    start_time = time.time()

    for _ in range(STEPS):
        # Signal worker to start
        flag.value = 1

        # Busy wait for worker to finish
        while flag.value == 1:
            pass

    end_time = time.time()

    # Cleanup
    flag.value = 2
    p.join()

    duration = end_time - start_time
    sps = STEPS / duration
    print(f"[Spinning Mode] Finished in {duration:.4f}s. Speed: {sps:.0f} steps/sec")

if __name__ == "__main__":
    print(f"PID: {os.getpid()} (Attach py-spy to this PID if running long experiments)")

    run_blocking_experiment()
    time.sleep(1) # Pause to separate traces visually
    run_spinning_experiment()


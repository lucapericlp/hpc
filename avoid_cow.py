"""
Using multiprocessing with NumPy and Shared Memory to avoid Copy-on-Write memory bloat.

Similar to the `mp_zerocopy.py` example but less concerned with the inter-process comms
latency but rather the RAM overhead caused by Copy-on-Write (CoW) when using standard
Python.
"""
import os
import sys
import time
import psutil
import concurrent.futures
import multiprocessing
from multiprocessing import shared_memory
import numpy as np

# Configuration
# 600 MB of data
DATA_SIZE_MB = 600
NUM_WORKERS = 4
WORKER_DURATION = 4

# Dimensions for our "Fake Image Data"
# We treat the 600MB as a list of 1KB chunks (simulating rows or small files)
CHUNK_SIZE = 1024
NUM_CHUNKS = (DATA_SIZE_MB * 1024 * 1024) // CHUNK_SIZE

# ---------------------------------------------------------
# SCENARIO A: STANDARD PYTHON LIST (The CoW Trap)
# ---------------------------------------------------------
def create_standard_data():
    print(f"[{os.getpid()}] Allocating {DATA_SIZE_MB}MB List of Byte strings...", flush=True)
    # A list of bytes.
    # Reading this in workers will increment ref-counts on the list items.
    return [b'\x01' * CHUNK_SIZE for _ in range(NUM_CHUNKS)]

def worker_standard(idx):
    # Simulate processing (Read-Only)
    count = 0
    # Process a stride to touch many pages
    stride = 100

    # Iterate through the list
    for i in range(0, len(shared_data_standard), stride):
        # accessing the item increments its ref-count -> Copy-on-Write Triggered!
        val = shared_data_standard[i]
        count += len(val)

    time.sleep(WORKER_DURATION)
    return count

# ---------------------------------------------------------
# SCENARIO B: NUMPY + SHARED MEMORY (The Zero-Copy Fix)
# ---------------------------------------------------------
def create_shm_data():
    print(f"[{os.getpid()}] Allocating {DATA_SIZE_MB}MB Shared NumPy Array...", flush=True)

    # 1. Calculate bytes needed
    # We use uint8 to match the "bytes" structure of the standard test
    shape = (NUM_CHUNKS, CHUNK_SIZE)
    dtype = np.uint8
    size_bytes = int(np.prod(shape)) * dtype().itemsize

    # 2. Allocate Shared Memory Block
    shm = shared_memory.SharedMemory(create=True, size=size_bytes)

    # 3. Create NumPy Array Wrapper
    # This array acts like a normal numpy array, but its memory is in 'shm'
    arr = np.ndarray(shape, dtype=dtype, buffer=shm.buf)

    # 4. Fill with dummy data
    arr[:] = 1 # Set all bytes to 1

    print(f"[{os.getpid()}] NumPy Array created in SHM: {shm.name}")
    return shm, shape, dtype

def worker_shm(shm_name, shape, dtype, idx):
    # 1. Attach to existing Shared Memory
    existing_shm = shared_memory.SharedMemory(name=shm_name)

    # 2. Create NumPy wrapper
    # This costs effectively 0 RAM (it's just a view)
    arr = np.ndarray(shape, dtype=dtype, buffer=existing_shm.buf)

    # 3. Perform Vectorized Work
    # Since 'arr' is backed by shared memory, and numpy releases the GIL
    # for heavy ops, this is incredibly efficient.
    # Crucially: NumPy does NOT use Python reference counting for the data elements!

    # Let's sum a stride just like the standard worker
    stride = 100
    # Slicing in numpy is a "View", not a copy. Very efficient.
    subset = arr[::stride]
    total = np.sum(subset)

    time.sleep(WORKER_DURATION)

    # 4. Cleanup
    existing_shm.close()
    return total

# ---------------------------------------------------------
# BENCHMARK ENGINE
# ---------------------------------------------------------
def run_benchmark(mode):
    global shared_data_standard # Only used for standard mode

    # Setup
    import gc
    gc.collect()
    time.sleep(0.5)
    mem_before = psutil.virtual_memory().available

    shm_handle = None
    shm_details = None

    # Allocation
    if mode == 'standard':
        shared_data_standard = create_standard_data()
    else:
        shm_handle, shape, dtype = create_shm_data()
        shm_details = (shm_handle.name, shape, dtype)

    # Measure Parent Cost
    time.sleep(1)
    mem_after_alloc = psutil.virtual_memory().available
    parent_cost = mem_before - mem_after_alloc

    # Spawn Workers
    ctx = multiprocessing.get_context('fork')
    print(f"[{os.getpid()}] Spawning {NUM_WORKERS} workers ({mode})...")

    min_available_mem = mem_after_alloc

    with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_WORKERS, mp_context=ctx) as executor:
        if mode == 'standard':
            futures = [executor.submit(worker_standard, i) for i in range(NUM_WORKERS)]
        else:
            # Pass metadata (Name, Shape, Dtype) to worker so it can reconstruct the array
            name, shape, dtype = shm_details
            futures = [executor.submit(worker_shm, name, shape, dtype, i) for i in range(NUM_WORKERS)]

        # Monitor Loop
        while not all(f.done() for f in futures):
            current_avail = psutil.virtual_memory().available
            if current_avail < min_available_mem:
                min_available_mem = current_avail
            time.sleep(0.1)

    # Cleanup SHM
    if shm_handle:
        shm_handle.close()
        shm_handle.unlink()

    # Results
    total_consumption = mem_before - min_available_mem
    overhead = total_consumption - parent_cost
    return parent_cost, overhead

if __name__ == "__main__":
    if sys.platform not in ('linux', 'darwin'):
        print("Error: Requires Linux/macOS for os.fork()")
        sys.exit(1)

    # MASTER CONTROL
    if len(sys.argv) == 1:
        import subprocess
        print(f"=== NumPy Shared Memory vs Standard List (Target: {DATA_SIZE_MB}MB) ===")
        print("Note: Monitoring system-wide available RAM.\n")

        # Run 1: Standard
        p1 = subprocess.run([sys.executable, __file__, "standard"], capture_output=True, text=True)
        print(p1.stdout)
        # Parse output
        try:
            base_p, base_o = map(float, p1.stderr.strip().split(','))
        except ValueError:
            print("Standard run failed or produced no output.")
            sys.exit(1)

        # Run 2: NumPy SHM
        p2 = subprocess.run([sys.executable, __file__, "numpy_shm"], capture_output=True, text=True)
        print(p2.stdout)
        try:
            opt_p, opt_o = map(float, p2.stderr.strip().split(','))
        except ValueError:
            print("NumPy run failed or produced no output.")
            sys.exit(1)

        print("\n=== FINAL RESULTS ===")
        print(f"{'Metric':<20} | {'Standard List':<15} | {'NumPy + SHM':<15}")
        print("-" * 60)
        print(f"{'Parent Allocation':<20} | {base_p/1024**2:5.0f} MB        | {opt_p/1024**2:5.0f} MB")
        print(f"{'Worker Overhead':<20} | {base_o/1024**2:5.0f} MB (High) | {opt_o/1024**2:5.0f} MB (Zero)")
        print("-" * 60)

        saved = base_o - opt_o
        print(f"RAM Saved: {saved/1024**2:.0f} MB")

    # WORKER CONTROL
    else:
        try:
            p_cost, overhead = run_benchmark(mode=sys.argv[1])
            sys.stderr.write(f"{p_cost},{overhead}")
        except Exception as e:
            import traceback
            traceback.print_exc()
            sys.exit(1)

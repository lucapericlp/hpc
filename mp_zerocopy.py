"""
Zero-Copy Shared Memory Demo with Multiprocessing

Using multiprocessing & numpy views to avoid costly data copies
"""
import multiprocessing
import numpy as np
import os
import ctypes

# ==========================================
# CONFIGURATION
# ==========================================
NUM_WORKERS = 4           # Number of parallel processes
ENVS_PER_WORKER = 2       # Environments simulated per process
OBS_DIM = 4               # Dimensions of a single observation (e.g., [x, y, vx, vy])
DTYPE = ctypes.c_float    # Underlying C type
NP_DTYPE = np.float32     # Matching NumPy type

# Total batch size (The "Super Batch")
TOTAL_ENVS = NUM_WORKERS * ENVS_PER_WORKER
FLAT_SIZE = TOTAL_ENVS * OBS_DIM

def worker_logic(shared_mem_raw, worker_id, start_flat_idx, end_flat_idx):
    """
    Simulates a Worker.

    Key Concept:
    The worker receives a reference to the HUGE shared memory block,
    but it only 'looks' at its specific slice.
    """
    # 1. WRAP: Create a numpy view of the WHOLE raw memory.
    #    This is instant (microseconds) and allocates NO new memory.
    full_buffer_view = np.frombuffer(shared_mem_raw, dtype=NP_DTYPE)

    # 2. SLICE: Narrow down to just this worker's section.
    #    "buffer=" in numpy lets us point to existing bytes.
    my_slice_flat = full_buffer_view[start_flat_idx:end_flat_idx]

    # 3. RESHAPE: Make it look like a batch of environments (N, D)
    my_batch_view = my_slice_flat.reshape((ENVS_PER_WORKER, OBS_DIM))

    # Verify pointer arithmetic (Pedagogical check)
    # The data pointer of this view should be offset from the base
    # offset = worker_id * ENVS_PER_WORKER * OBS_DIM * 4 bytes
    pointer = my_batch_view.__array_interface__['data'][0]
    print(f"  [Worker {worker_id}] Writing to slice {start_flat_idx}-{end_flat_idx} (Ptr: ...{str(pointer)[-4:]})")

    # 4. WRITE: Modify data in-place.
    #    Because this maps to shared memory, the Main Process sees this INSTANTLY.
    #    No send(), no pickle, no pipe.
    for i in range(ENVS_PER_WORKER):
        # Generate distinct data: WorkerID.EnvID
        val = worker_id + (i * 0.1)
        my_batch_view[i, :] = val

def run_zerocopy_experiment():
    print(f"PID: {os.getpid()}")
    print(f"\n[Zero-Copy Demo] Allocating Super-Batch for {TOTAL_ENVS} environments...")

    # ==========================================
    # 1. ALLOCATE THE ARENA
    # ==========================================
    # This RawArray is pinned in shared memory.
    # It will hold observations for ALL agents across ALL workers.
    shared_mem_raw = multiprocessing.RawArray(DTYPE, FLAT_SIZE)

    # Create the Main Process View (The "Tensor" the GPU would see)
    main_batch_view = np.frombuffer(shared_mem_raw, dtype=NP_DTYPE).reshape(TOTAL_ENVS, OBS_DIM)

    main_ptr = main_batch_view.__array_interface__['data'][0]
    print(f"  [Main] Super-Batch Allocated at Ptr: ...{str(main_ptr)[-4:]}")
    print(f"  [Main] Initial Data (Should be zeros):\n{main_batch_view}\n")

    # ==========================================
    # 2. SLICE-BASED BATCHING (The Dispatch)
    # ==========================================
    processes = []

    print("[Main] Launching Workers...")
    for i in range(NUM_WORKERS):
        # Calculate exactly which bytes belong to this worker
        # This is the "Pattern" -> Deterministic slicing
        start_env_idx = i * ENVS_PER_WORKER
        end_env_idx   = (i + 1) * ENVS_PER_WORKER

        start_flat_idx = start_env_idx * OBS_DIM
        end_flat_idx   = end_env_idx   * OBS_DIM

        p = multiprocessing.Process(
            target=worker_logic,
            args=(shared_mem_raw, i, start_flat_idx, end_flat_idx)
        )
        processes.append(p)
        p.start()

    # Wait for workers to finish writing
    for p in processes:
        p.join()

    # ==========================================
    # 3. ZERO-COPY READ
    # ==========================================
    print("\n[Main] Workers Joined. Reading Super-Batch...")
    print("  (Note: We did NOT call pipe.recv() or queue.get())")

    # We just look at our local array, and the data is there.
    # This is the "magic" of shared memory.
    print("\nUpdated Super-Batch contents:")
    print(main_batch_view)

    # Verification
    print("\n[Verification]")
    # Check Env 0 (Worker 0)
    print(f"  Env 0 (Worker 0): {main_batch_view[0]} -> Expect [0. 0. 0. 0.]")

    # Check Env 3 (Worker 1, 2nd env) -> Worker ID 1 + 0.1 = 1.1
    # Global Index 3 corresponds to Worker 1, local index 1
    target_env = ENVS_PER_WORKER + 1
    print(f"  Env {target_env} (Worker 1): {main_batch_view[target_env]} -> Expect [1.1 1.1 1.1 1.1]")

if __name__ == "__main__":
    run_zerocopy_experiment()

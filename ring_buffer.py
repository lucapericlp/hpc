"""
Demonstrates a simple implementation of a ring buffer as used in audio processing
contexts.

$ python ring_buffer.py \
    --input-audio-file yo.wav \
    --output-audio-file output.wav \
    --no-consumer-jitter \
    --no-producer-jitter

Ring Buffer: https://en.wikipedia.org/wiki/Ring_buffer
"""

import threading
import time
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import tyro


class RingBuffer:
    """
    A Thread-Safe 1D Ring Buffer (Circular Buffer) implementation using NumPy.
    """

    def __init__(self, capacity: int, dtype: type = np.float32):
        self.capacity = capacity
        self.buffer = np.zeros(capacity, dtype=dtype)
        self._write_cursor = 0
        self._read_cursor = 0
        self._lock = threading.Lock()  # Mutex for thread safety

    def write(self, data: np.ndarray) -> int:
        """
        Writes data to the buffer in a thread-safe manner.
        Overwrites oldest data if full.
        """
        with self._lock:
            n_to_write = len(data)

            # Clamp to capacity if input is massive
            if n_to_write > self.capacity:
                data = data[-self.capacity :]
                n_to_write = self.capacity

            start_idx = self._write_cursor % self.capacity
            end_idx = (self._write_cursor + n_to_write) % self.capacity

            if end_idx > start_idx:
                self.buffer[start_idx:end_idx] = data[:n_to_write]
            else:
                part1_len = self.capacity - start_idx
                self.buffer[start_idx:] = data[:part1_len]
                self.buffer[:end_idx] = data[part1_len:n_to_write]

            self._write_cursor += n_to_write

            # Handle overlap/overwrite
            if self._write_cursor - self._read_cursor > self.capacity:
                self._read_cursor = self._write_cursor - self.capacity

            return n_to_write

    def read(self, count: int) -> np.ndarray:
        """
        Reads 'count' elements from the buffer in a thread-safe manner.
        """
        with self._lock:
            available = self._write_cursor - self._read_cursor
            n_to_read = min(count, available)

            if n_to_read == 0:
                return np.array([], dtype=self.buffer.dtype)

            start_idx = self._read_cursor % self.capacity
            end_idx = (self._read_cursor + n_to_read) % self.capacity

            out_data = np.empty(n_to_read, dtype=self.buffer.dtype)

            if end_idx > start_idx:
                out_data[:] = self.buffer[start_idx:end_idx]
            else:
                part1_len = self.capacity - start_idx
                out_data[:part1_len] = self.buffer[start_idx:]
                out_data[part1_len:] = self.buffer[:end_idx]

            self._read_cursor += n_to_read
            return out_data

    @property
    def fill_level(self) -> int:
        """Thread-safe access to fill level."""
        with self._lock:
            return self._write_cursor - self._read_cursor


def producer_task(
    wav: np.ndarray,
    rb: RingBuffer,
    sample_rate: int,
    chunk_size: int,
    stop_event: threading.Event,
    simulate_jitter: bool,
):
    """Generates audio data and pushes it to the ring buffer."""
    t = 0.0
    total_samples = wav.shape[0]
    print(
        f"[Producer] Loaded audio file with {total_samples} samples at {sample_rate} Hz."
    )
    written_samples = 0

    print("[Producer] Started.")
    while written_samples < total_samples and not stop_event.is_set():
        # Simulate real-time audio production timing
        # In a real system, this is driven by hardware interrupts/callbacks
        rtf = 1.0
        if simulate_jitter:
            rtf += np.random.uniform(-0.5, -0.1)

        time.sleep(chunk_size / (sample_rate * rtf))

        data_chunk = wav[written_samples : written_samples + chunk_size]

        rb.write(data_chunk)

        t += chunk_size / sample_rate
        written_samples += chunk_size

        # Occasional log
        if (written_samples // chunk_size) % 10 == 0:
            print(f"[Producer] Buffer Fill: {rb.fill_level}/{rb.capacity}")

    print("[Producer] Finished generating data.")


def consumer_task(
    output_path: Path,
    rb: RingBuffer,
    sample_rate: int,
    chunk_size: int,
    stop_event: threading.Event,
    simulate_jitter: bool,
):
    """Reads audio data from the ring buffer and processes it."""
    print("[Consumer] Started.")

    # 1. Buffering (Pre-roll)
    # Wait until buffer has enough data to start "playback"
    # This prevents immediate underrun due to startup latency
    pre_roll_chunks = 4
    print(f"[Consumer] Buffering {pre_roll_chunks} chunks...")
    while rb.fill_level < chunk_size * pre_roll_chunks and not stop_event.is_set():
        time.sleep(0.01)
    print("[Consumer] Buffering complete. Starting playback.")

    with sf.SoundFile(
        output_path, mode="w", samplerate=sample_rate, channels=1, subtype="PCM_16"
    ) as _output_sf:
        while not stop_event.is_set():
            # Simulate Real-Time Consumption
            # A real DAC consumes samples at exactly the sample rate.
            # We sleep for the duration of the chunk to simulate this.
            # Processing time (0.005) is included in this "frame time" conceptually,
            # but to keep it simple, we just sleep the frame duration.
            time.sleep(chunk_size / sample_rate)

            read_data = rb.read(chunk_size)

            if len(read_data) > 0:
                # append to file
                _output_sf.write(read_data)

                # Optional jitter simulation (Consumer falls behind)
                if simulate_jitter:
                    rtf = 1.0 - np.random.uniform(0.1, 0.5)
                    print("[Consumer] Jitter glitch!")
                    time.sleep(chunk_size / (sample_rate * rtf))
            else:
                print("[Consumer] Buffer empty (Underrun), emitting silence...")
                _output_sf.write(np.zeros(chunk_size, dtype=np.float32))
                # If we underrun, we are already "late", so we loop immediately
                # (or sleep a tiny bit to avoid CPU spin if completely empty for long time)

    print("[Consumer] Stopped.")


def stream_audio(
    input_audio_file: Path = Path("yo.wav"),
    output_audio_file: Path = Path("output.wav"),
    consumer_jitter: bool = False,
    producer_jitter: bool = False,
):
    print("Initializing Threaded Ring Buffer Simulation...")

    SAMPLE_RATE = 22050
    BUFFER_SIZE = 4096  # Larger buffer for threaded stability
    CHUNK_SIZE = 512

    rb = RingBuffer(capacity=BUFFER_SIZE, dtype=np.float32)
    stop_event = threading.Event()

    wav, _ = librosa.load(input_audio_file, sr=SAMPLE_RATE, mono=True)
    wav = np.concatenate([wav, wav, wav])  # Extend for demo

    # Create threads
    prod_thread = threading.Thread(
        target=producer_task,
        args=(wav, rb, SAMPLE_RATE, CHUNK_SIZE, stop_event, producer_jitter),
    )
    cons_thread = threading.Thread(
        target=consumer_task,
        args=(
            output_audio_file,
            rb,
            SAMPLE_RATE,
            CHUNK_SIZE,
            stop_event,
            consumer_jitter,
        ),
    )

    # Start threads
    prod_thread.start()
    cons_thread.start()

    try:
        # Let them run until producer finishes
        prod_thread.join()

        # Give consumer a moment to drain remaining data
        time.sleep(0.2)
        print("[Main] Stopping consumer...")
        stop_event.set()
        cons_thread.join()

    except KeyboardInterrupt:
        print("\n[Main] Interrupted! Stopping threads...")
        stop_event.set()
        prod_thread.join()
        cons_thread.join()

    print("Simulation Complete.")


if __name__ == "__main__":
    tyro.cli(stream_audio)

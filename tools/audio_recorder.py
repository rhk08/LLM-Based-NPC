import pyaudio
import wave
import threading
import os

class AudioRecorder:
    def __init__(self, save_dir="./recordings", filename="output.wav"):
        # Audio recording parameters
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 44100
        self.CHUNK = 1024

        self.SAVE_DIR = save_dir
        self.FILENAME = filename
        self.OUTPUT_FILENAME = os.path.join(self.SAVE_DIR, self.FILENAME)

        # Ensure the directory exists
        if not os.path.exists(self.SAVE_DIR):
            os.makedirs(self.SAVE_DIR)

        # Initialize pyaudio
        self.audio = pyaudio.PyAudio()
        self.recording = False
        self.frames = []
        self.stream = None

    def start_recording(self):
        self.recording = True
        self.frames = []
        self.stream = self.audio.open(format=self.FORMAT, channels=self.CHANNELS, rate=self.RATE,
                                      input=True, frames_per_buffer=self.CHUNK)
        print("Recording started...")
        recording_thread = threading.Thread(target=self.record)
        recording_thread.start()

    def record(self):
        while self.recording:
            data = self.stream.read(self.CHUNK)
            self.frames.append(data)

    def stop_recording(self):
        self.recording = False
        self.stream.stop_stream()
        self.stream.close()
        with wave.open(self.OUTPUT_FILENAME, 'wb') as wf:
            wf.setnchannels(self.CHANNELS)
            wf.setsampwidth(self.audio.get_sample_size(self.FORMAT))
            wf.setframerate(self.RATE)
            wf.writeframes(b''.join(self.frames))
        print(f"Recording saved to {self.OUTPUT_FILENAME}")

    def cleanup(self):
        self.audio.terminate()

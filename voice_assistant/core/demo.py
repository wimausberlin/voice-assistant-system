from inference import CNNNetworkInference
from typing import List, Tuple

import os
import pyaudio
import time
import torchaudio
import wave

CHANNELS = 1
CHUNK = 1024
FORMAT = pyaudio.paInt16
SAMPLE_RATE = 8000
FILENAME='demo.wav'


def listerner(record_seconds: int = 2, filename: str = FILENAME) -> None:
    input("Press enter to record.")
    time.sleep(0.2)
    p = pyaudio.PyAudio()
    stream = p.open(
        rate=SAMPLE_RATE,
        channels=CHANNELS,
        format=FORMAT,
        input=True,
        frames_per_buffer=CHUNK,
        input_device_index=None,
    )
    frames = []
    for i in range(SAMPLE_RATE // CHUNK * record_seconds):
        data = stream.read(CHUNK)
        frames.append(data)
    stream.stop_stream()
    stream.close()
    p.terminate()

    # save the audio file
    wf = wave.open(filename, "wb")  # wb for 'write bytes' mode
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(SAMPLE_RATE)
    wf.writeframes(b"".join(frames))
    wf.close()


def main()->None:
    listerner()
    signal, sr = torchaudio.load(FILENAME)
    inference = CNNNetworkInference()
    pred = inference.get_prediction(signal)
    if pred == 1:
        print("Detected")
    else:
        print("Not detected")
    os.remove(FILENAME)


if __name__ == "__main__":
    main()

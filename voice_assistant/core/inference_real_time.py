from typing import Tuple
import numpy as np
import pyaudio
import sys
import time

from core.inference import CNNNetworkInference
from queue import Queue
from threading import Thread


CHANNEL=1
FORMAT=pyaudio.paInt16
SECONDS=2
SAMPLE_RATE=8000
SLIDING_WINDOW_SECS=1/8
RUN=True

CHUNK = int(SLIDING_WINDOW_SECS*SAMPLE_RATE*SECONDS)

def get_audio_input_stream(callback)->pyaudio.PyAudio:
    stream = pyaudio.PyAudio().open(
        format=FORMAT,
        channels=CHANNEL,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=CHUNK,
        input_device_index=0,
        stream_callback=callback)
    return stream

def callback(in_data:np.array,data:np.array,q:Queue,timeout:time,feed_samples:int,silence_threshold:int=100)->Tuple[np.array,pyaudio.PyAudio]:
    global RUN
    if time.time() > timeout:
        RUN = False        
    data0 = np.frombuffer(in_data, dtype='int16')
    if np.abs(data0).mean() < silence_threshold:
        sys.stdout.write('-')
        return (in_data, pyaudio.paContinue)
    else:
        sys.stdout.write('.')
    data = np.append(data,data0)    
    if len(data) > feed_samples:
        data = data[-feed_samples:]
        # Process data async by sending a queue.
        q.put(data)
    return (in_data, pyaudio.paContinue)

def main()->None:
    global RUN
    inference=CNNNetworkInference()
    #Queue to communiate between the audio callback and main thread
    q = Queue()

    # Run the demo for a timeout seconds
    timeout = time.time() + 1 #1sec

    feed_samples=SAMPLE_RATE*SECONDS
    # Data buffer for the input wavform
    data = np.zeros(feed_samples, dtype='int16')    
    stream = get_audio_input_stream(callback)
    stream.start_stream()
    try:
        while RUN:
            data = q.get()
            new_trigger = inference.get_prediction(data)
            if new_trigger==1:
                print('activate')
    except (KeyboardInterrupt, SystemExit):
        stream.stop_stream()
        stream.close()
        RUN = False

    stream.stop_stream()
    stream.close()
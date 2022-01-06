""" 
Collect the audio data for the binary classification training ()
:arg sampling_rate: rate sampling, default is at 8000
:arg seconds: how many second to collect environment or your own wake word sound, default is at None. If None, records indefinitely
:arg save_path: path to save the environment sound (eg. sound/environment.wav), default is at None.
:arg sample_save_path: path to save the word samples (eg. sound/), default is at None.
:arg samples: set to the interactive mode (if you want to record your word multiple times), default None.

Step 1. Collecting environment sound : run code with --seconds x else this will record indefinitely until ctrl + c 
Step 2. Collecting wake word samples : run code with --samples_save_path and --sample 

Some parts of this code is from Michael Nguyen :
https://github.com/LearnedVector/A-Hackers-AI-Voice-Assistant.git
"""
import argparse
import os
import pyaudio
import time
import wave

from typing import List, Tuple

"""
CHUNK is the number of frames in the buffer.
Each frame will have 1 sample as "CHANNELS=1".
"""

CHANNELS = 1
CHUNK = 1024
FORMAT = pyaudio.paInt16


class Listener:
    def __init__(
        self,
        args: Tuple[int, int, str, str, bool, bool],
        input_device_index: int = None,
    ) -> None:
        """
        :args sample_rate: the number of samples collected per second.
        :args seconds: the number of seconds recorded.
        :args save_path: full path to save file. i.e. sound.wav
        :args samples_save_path: directory to save all the word samples (i.e. sound/)
        :args sample: sets to interactive mode
        :args input_index: get the input_index
        :param input_device_index: allow to open the micro and stream with a specific input device index
        """
        self.channels = CHANNELS
        self.chunk = CHUNK
        self.FORMAT = FORMAT
        self.sample_rate = args.sample_rate
        self.record_seconds = args.seconds

        self.p = pyaudio.PyAudio()

        self.stream = self.p.open(
            rate=self.sample_rate,
            channels=self.channels,
            format=self.FORMAT,
            input=True,
            frames_per_buffer=self.chunk,
            input_device_index=input_device_index,
        )

    def save_audio(self, filename: str, frames: List[str]) -> None:
        """
        :param filename: name of the saved new audio file.
        :param frames: the frames of data.
        """
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()

        # save the audio file
        wf = wave.open(filename, "wb")  # wb for 'write bytes' mode
        wf.setnchannels(self.channels)
        wf.setsampwidth(self.p.get_sample_size(self.FORMAT))
        wf.setframerate(self.sample_rate)
        wf.writeframes(b"".join(frames))
        wf.close()

    def get_frames(self) -> List[str]:
        frames = []
        print(self.sample_rate // self.chunk * self.record_seconds)
        for i in range(self.sample_rate // self.chunk * self.record_seconds):
            data = self.stream.read(self.chunk)
            frames.append(data)
        return frames


def next_path_index(path_pattern: str) -> int:
    """
    Finds the next free path index in an sequentially named list of files

    e.g. path_pattern = 'sound%s.wav':

    sound1.wav
    sound2.wav
    sound3.wav
    """
    i = 1
    # First do an exponential search
    while os.path.exists(path_pattern % i):
        i = i * 2
    # Result lies somewhere in the interval (i/2..i]
    # We call this interval (a..b] and narrow it down until a + 1 = b
    a, b = (i // 2, i)
    while a + 1 < b:
        c = (a + b) // 2  # interval midpoint
        a, b = (c, b) if os.path.exists(path_pattern % c) else (a, c)
    return b


def sample(args: Tuple[int, int, str, str, bool, bool]) -> None:
    idx = next_path_index(args.samples_save_path + "%s.wav")
    try:
        input("Press enter to proceed recording \n")
        while True:
            listerner = Listener(args)
            frames = []
            input(f"Press enter to record. Each sample will be {args.seconds} seconds.")
            time.sleep(0.2)  # allow to not understand the enter clic
            frames = listerner.get_frames()
            save_path = os.path.join(args.samples_save_path, f"{idx}.wav")
            listerner.save_audio(save_path, frames)
            idx += 1
    except KeyboardInterrupt:
        print("Keyboard Interrupt")
    except Exception as e:
        print(str(e))


def get_input_index() -> None:
    p = pyaudio.PyAudio()
    info = p.get_host_api_info_by_index(0)
    numdevices = info.get("deviceCount")
    for i in range(0, numdevices):
        if (
            p.get_device_info_by_host_api_device_index(0, i).get("maxInputChannels")
        ) > 0:
            print(
                "Input Device id ",
                i,
                " - ",
                p.get_device_info_by_host_api_device_index(0, i).get("name"),
            )


def main(args: Tuple[int, int, str, str, bool, bool]) -> None:
    listener = Listener(args)
    frames = []
    try:
        while True:
            if listener.record_seconds == None:
                print("Recording indefinitely. Press ctrl+c to quit.", end="\r")
                frames = [listener.stream.read(listener.chunk)]
            else:
                frames = listener.get_frames()
                raise Exception("done recording")
    except KeyboardInterrupt:
        print("Keyboard Interrupt")
    except Exception as e:
        print(str(e))
    listener.save_audio(args.save_path, frames)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""
        Script to collect audio data for the binary classification training.
        To record environment sound run script only with --save_path arg. This will
        record indefinitely until ctrl + c.
        To record for a set amount of time set --seconds to whatever you want.
        To record your own word, use --interactive mode with the --sample_save_path arg.\n
        Step 1. Collecting environment sound : run code with --seconds x else this will record indefinitely until ctrl + c
        Step 2. Collecting wake word samples : run code with --samples_save_path and --sample

        To have more information about the input index, use --input_index
        """,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--sample_rate",
        type=int,
        default=8000,
        help="the number of samples collected per second, default at 8000",
    )
    parser.add_argument(
        "--seconds",
        type=int,
        default=None,
        help="how many second to collect environment or your own wake word sound, default is at None. If None, then will record forever until keyboard interrupt",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default=None,
        required=False,
        help="full path to save file. i.e. sound.wav",
    )
    parser.add_argument(
        "--samples_save_path",
        type=str,
        default=None,
        required=False,
        help="directory to save all the word samples (i.e. sound/)",
    )
    parser.add_argument(
        "--sample",
        default=False,
        action="store_true",
        required=False,
        help="sets to interactive mode",
    )
    parser.add_argument(
        "--input_index",
        default=False,
        action="store_true",
        required=False,
        help="get the input_index",
    )

    args = parser.parse_args()
    if args.input_index:
        get_input_index()
    else:
        if args.sample:
            if args.samples_save_path is None:
                raise Exception("need to set --samples_save_path")
            sample(args)
        else:
            if args.save_path is None:
                raise Exception("need to set --save_path")
            main(args)

import torchaudio
import torchaudio.transforms as T
import time

from pandas.io import json
from core import dataset


file_path = "../../sound/1/"
file_path_sound = file_path+"2.wav"


def test_mfcc()->None:
    waveform, sample_rate = torchaudio.load(file_path_sound)
    mfcc_tranf = T.MFCC(sample_rate)
    mfcc = mfcc_tranf(waveform)
    dataset.plot_spectrogram(mfcc[0], "MFCC")


def test_spectAugment_mfcc()->None:
    spectA_tranf = dataset.SpectAugment()
    waveform, sample_rate = torchaudio.load(file_path_sound)
    mfcc_tranf = T.MFCC(sample_rate)
    mfcc = mfcc_tranf(waveform)
    spectA = spectA_tranf(mfcc)
    dataset.plot_spectrogram(spectA[0], "SpectAugment MFCC")

def initWakeWordDataset()->None:
    json_path="../../train.json"
    datasetWakeWord=dataset.WakeWordDataset(json_path)
    print(json_path)

#test_mfcc()
#test_spectAugment_mfcc()
#initWakeWordDataset()
#listen_audio()

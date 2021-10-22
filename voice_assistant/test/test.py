from pandas.io import json
from core import dataset

import os
import pandas as pd
import sys
import torchaudio
import torchaudio.transforms as T


file_path = "../../sound/1/"
file_path_sound = file_path+"2.wav"


def test_mfcc():
    waveform, sample_rate = torchaudio.load(file_path_sound)
    mfcc_tranf = T.MFCC(sample_rate)
    mfcc = mfcc_tranf(waveform)
    dataset.plot_spectrogram(mfcc[0], "MFCC")


def test_spectAugment_mfcc():
    spectA_tranf = dataset.SpectAugment()
    waveform, sample_rate = torchaudio.load(file_path_sound)
    mfcc_tranf = T.MFCC(sample_rate)
    mfcc = mfcc_tranf(waveform)
    spectA = spectA_tranf(mfcc)
    dataset.plot_spectrogram(spectA[0], "SpectAugment MFCC")


# test_mfcc()
# test_spectAugment_mfcc()

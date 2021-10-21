"""
Some parts of this code is from Michael Nguyen :
https://github.com/LearnedVector/A-Hackers-AI-Voice-Assistant.git
"""

from scipy.fft._pocketfft.helper import _normalization
from sonopy import power_spec, mel_spec, mfcc_spec, filterbanks
from torch.functional import Tensor
from torch.nn import Module, Sequential
from torch.utils.data import Dataset
from typing import List, Tuple

import librosa
import matplotlib
import matplotlib.pyplot as plt
import os
import pandas as pd
import torch
import torchaudio
import torchaudio.transforms as T


class RandomCut(Module):
    """Augmentation technique that randomly cuts start or end of the audio"""

    def __init__(self, max_cut: int = 10):
        super(RandomCut, self).__init__()
        self.max_cut = max_cut

    def forward(self, x: List[T.MFCC]) -> List[T.MFCC]:
        """Cuts some random part of the MFCC"""
        side = torch.randint(0, 2, (1,))
        cut = torch.randint(1, self.max_cut+1, (1,))
        if side == 0:
            return x[:-cut, :, :]
        else:
            return x[cut:, :, :]


class SpectAugment(Module):
    def __init__(self, freq_mask: int = 3, time_mask: int = 3) -> None:
        super(SpectAugment, self).__init__()
        self.specaug = Sequential(
            T.FrequencyMasking(freq_mask_param=freq_mask),
            T.TimeMasking(time_mask_param=time_mask)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.specaug(x)


class WakeWordData(Dataset):
    """Process wakeword data"""

    def __init__(self, json_path: str, sample_rate: int = 8000, valid: bool = False) -> None:
        super(WakeWordData, self).__init__()
        self.sample_rate = sample_rate
        self.data = pd.read_json(json_path,lines=True)
        if valid:
            self.audio_transform = T.MFCC(sample_rate)
        else:
            self.audio_transform = Sequential(
                T.MFCC(sample_rate),
                SpectAugment()
            )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.tensor, int]:
        if torch.is_tensor(idx):
            idx = idx.item()
        file_path = self.data.key.iloc[idx]
        waveform, sample_rate = torchaudio.load(file_path)
        if sample_rate>self.sample_rate:
            waveform = T.Resample(sample_rate, self.sample_rate)(waveform)
        mfcc = self.audio_transform(waveform)
        label = self.data.label.iloc[idx]
        return mfcc, int(label)


def collate_fn(data: List[Tuple[torch.Tensor, int]]) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    mfccs, labels = [], []
    for d in data:
        mfcc, label = d
        mfccs.append(mfcc.squeeze(0).transpose(0, 1))
        labels.append(label)

    mfccs = torch.nn.utils.rnn.pad_sequence(mfccs, batch_first=True)
    mfccs = mfccs.transpose(0, 1)
    labels = torch.Tensor(labels)
    return mfccs, labels

def plot_spectrogram(spec, title=None, ylabel='freq_bin', aspect='auto', xmax=None):
  fig, axs = plt.subplots(1, 1)
  axs.set_title(title or 'Spectrogram (db)')
  axs.set_ylabel(ylabel)
  axs.set_xlabel('frame')
  im = axs.imshow(librosa.power_to_db(spec), origin='lower', aspect=aspect)
  if xmax:
    axs.set_xlim((0, xmax))
  fig.colorbar(im, ax=axs)
  plt.show()


def test_mfcc(file_path):
    waveform, sample_rate = torchaudio.load(file_path)
    mfcc_tranf = T.MFCC(sample_rate)
    mfcc=mfcc_tranf(waveform)
    plot_spectrogram(mfcc[0])
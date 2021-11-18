"""
Some parts of this code is from Michael Nguyen :
https://github.com/LearnedVector/A-Hackers-AI-Voice-Assistant.git
"""

from scipy.fft._pocketfft.helper import _normalization
from sonopy import power_spec, mel_spec, mfcc_spec, filterbanks
from torch.functional import Tensor
from torch.nn import Module, Sequential
from torch.utils import data
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
        self.data = pd.read_json(json_path, lines=True)
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
        if sample_rate > self.sample_rate:
            waveform = T.Resample(sample_rate, self.sample_rate)(waveform)
        mfcc = self.audio_transform(waveform)
        label = self.data.label.iloc[idx]
        return mfcc, int(label)


class WakeWordDataset(Dataset):
    def __init__(self, json_path: str, sample_rate: int = 8000, num_samples: int = 16000, audio_transformation: T = T.MelSpectrogram, device:str="cpu") -> None:
        super().__init__()
        self.data = pd.read_json(json_path, lines=True)
        self.sample_rate = sample_rate
        self.num_samples = num_samples
        self.device=device
        self.audio_transformation = audio_transformation.to(device)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Tuple[torch.tensor, str]:
        file_path = self.data.key.iloc[index]
        label = self.data.label.iloc[index]
        signal, sample_rate = torchaudio.load(file_path)
        if sample_rate > self.sample_rate:
            signal = T.Resample(sample_rate, self.sample_rate)(signal)
        if signal.shape[1] > self.num_samples:
            signal = signal[:, :self.num_samples]
        elif signal.shape[1] < self.num_samples:
            num_missing_samples = self.num_samples-signal.shape[1]
            last_dim_padding = (0, num_missing_samples)
            signal = torch.nn.functional.pad(signal, last_dim_padding)
        signal = self.audio_transformation(signal)
        return signal, label


def collate_fn(data: List[Tuple[torch.Tensor, int]]) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    mfccs, labels = [], []
    for d in data:
        mfcc, label = d
        mfccs.append(mfcc.squeeze(0).transpose(0, 1))
        labels.append(label)

    mfccs = torch.nn.utils.rnn.pad_sequence(mfccs, batch_first=True)
    labels = torch.Tensor(labels)
    return mfccs, labels


def plot_spectrogram(spec, title: str = None, ylabel: str = 'freq_bin', aspect: str = 'auto', xmax: float = None):
    fig, axs = plt.subplots(1, 1)
    axs.set_title(title or 'Spectrogram (db)')
    axs.set_ylabel(ylabel)
    axs.set_xlabel('frame')
    im = axs.imshow(librosa.power_to_db(spec), origin='lower', aspect=aspect)
    if xmax:
        axs.set_xlim((0, xmax))
    fig.colorbar(im, ax=axs)
    plt.show()


def test_wakeWordData():
    data_f = WakeWordData("../../../train.json")
    data = [data_f.__getitem__(1)]
    m, l = collate_fn(data)
    plot_spectrogram(m[0])


def test_WakeWordDataset():
    json_path = "../../../train.json"
    transformation = T.MelSpectrogram(
        sample_rate=8000, n_fft=1024, hop_length=512, n_mels=64)
    datasetWakeWord = WakeWordDataset(
        json_path, audio_transformation=transformation)
    signal, label = datasetWakeWord.__getitem__(5)
    return signal, label


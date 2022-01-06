import librosa
import matplotlib
import matplotlib.pyplot as plt
import os
import torch
import torchaudio
import torchaudio.transforms as T


import numpy as np

import librosa.display

PATH="../../../sound/1/27.wav"

def print_stats(waveform:torch.Tensor, sample_rate:int=None, src:int=None)->None:
  if src:
    print("-" * 10)
    print("Source:", src)
    print("-" * 10)
  if sample_rate:
    print("Sample Rate:", sample_rate)
  print("Shape:", tuple(waveform.shape))
  print("Dtype:", waveform.dtype)
  print(f" - Max:     {librosa.power_to_db(waveform.max().item()):6.3f}")
  print(f" - Min:     {waveform.min().item():6.3f}")
  print(f" - Mean:    {waveform.mean().item():6.3f}")
  print(f" - Std Dev: {waveform.std().item():6.3f}")
  print()
  print(waveform)
  print()

def plot_waveform(waveform:torch.Tensor, sample_rate:int, title:str="Waveform", xlim:int=None, ylim:int=None)->None:
  waveform = waveform.numpy()

  num_channels, num_frames = waveform.shape
  time_axis = torch.arange(0, num_frames) / sample_rate

  figure, axes = plt.subplots(num_channels, 1)
  if num_channels == 1:
    axes = [axes]
  for c in range(num_channels):
    axes[c].plot(time_axis, waveform[c], linewidth=1)
    axes[c].grid(True)
    if num_channels > 1:
      axes[c].set_ylabel(f'Channel {c+1}')
    if xlim:
      axes[c].set_xlim(xlim)
    if ylim:
      axes[c].set_ylim(ylim)
  #figure.suptitle(title)
  plt.show()

def get_spectrogram(waveform:torch.Tensor,sample_rate:int)->T.MelSpectrogram:
  spectrogram = T.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )
  return spectrogram(waveform)

def plot_spectrogram(spec:torch.Tensor, title: str = None, ylabel: str = 'freq_bin', aspect: str = 'auto', xmax: float = None)->None:
    fig, axs = plt.subplots(1, 1)
    #axs.set_title(title or 'Spectrogram (db)')
    axs.set_ylabel(ylabel)
    axs.set_xlabel('frame')
    im = axs.imshow(librosa.power_to_db(spec), origin='lower', aspect=aspect,cmap='magma')
    if xmax:
        axs.set_xlim((0, xmax))
    fig.colorbar(im, ax=axs)
    plt.show()

def plot_librosa()->None:
  y, sr = librosa.load(PATH)
  D = librosa.stft(y)  # STFT of y
  S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
  plt.figure()
  librosa.display.specshow(S_db)
  plt.colorbar()
  plt.show()


def main(path:str=PATH)->None:
    waveform, sample_rate = torchaudio.load(PATH)
    spec=get_spectrogram(waveform,sample_rate)
    plot_spectrogram(spec[0],ylabel='mel freq')
    #plot_spectrogram(waveform,sample_rate)
    #print_stats(waveform,sample_rate)
    #plot_librosa()
    #plot_waveform(waveform,sample_rate)


if __name__=="__main__":
    main()
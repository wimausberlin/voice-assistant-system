import torch
import torchaudio.transforms as T

from dataset import WakeWordDataset
from model import CNNNetwork
from typing import Tuple


@torch.no_grad()
def predict(model, input:torch.Tensor, class_mapping:Tuple[int])->int:
    model.eval()
    predictions = model(input)
    predicted_index = predictions[0].argmax(0)
    predicted = class_mapping[predicted_index]
    return predicted

class CNNNetworkInference:
    def __init__(self,class_mapping:Tuple[int]=[0, 1]) -> None:
        self.model_cnn = CNNNetwork()
        self.state_dict = torch.load("model_cnn.pth")
        self.model_cnn.load_state_dict(self.state_dict)
        self.class_mapping=class_mapping

        self.mel_spectrogram = T.MelSpectrogram(
            sample_rate=8000,
            n_fft=1024,
            hop_length=512,
            n_mels=64
        )
    def get_prediction(self,x:torch.Tensor)->int:
        if x.shape[1] > 16000:
            x = x[:, :16000]
        elif x.shape[1] < 16000:
            num_missing_samples = 16000-x.shape[1]
            last_dim_padding = (0, num_missing_samples)
            x = torch.nn.functional.pad(x, last_dim_padding)
        mel_spectro=self.mel_spectrogram(x)
        mel_spectro.unsqueeze_(0)
        pred=predict(self.model_cnn,mel_spectro,self.class_mapping)
        return pred

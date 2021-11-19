from torch.utils.data import dataset
from dataset import WakeWordDataset
from model import CNNNetwork

import torch
import torchaudio.transforms as T

class_mapping = [0, 1]
test_data_json = "../../../test.json"


@torch.no_grad()
def predict(model, input, class_mapping):
    model.eval()
    predictions = model(input)
    predicted_index = predictions[0].argmax(0)
    predicted = class_mapping[predicted_index]
    return predicted


def main():
    pass


if __name__ == "__main__":
    model_cnn = CNNNetwork()
    state_dict = torch.load("model_cnn.pth")
    model_cnn.load_state_dict(state_dict)

    mel_spectrogram = T.MelSpectrogram(
        sample_rate=8000,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )

    test_dataset = WakeWordDataset(
        test_data_json, audio_transformation=mel_spectrogram, device="cpu")
    for i in range(4):
        # [batch size, num_channels, fr, time]
        input, label = test_dataset[i][0], test_dataset[i][1]
        input.unsqueeze_(0)

        predicted = predict(model_cnn, input, class_mapping)

        print(f"Predicted: '{predicted}', expected: '{label}'")

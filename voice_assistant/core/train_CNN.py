from ast import parse
from torch.functional import Tensor

from torch.nn.modules.loss import BCELoss, CrossEntropyLoss
from dataset import WakeWordDataset
from model import CNNNetwork
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Tuple

import argparse
import torch
import torchaudio.transforms as T
import torch.nn as nn
import torch.optim as optim

def binary_accuracy(preds: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # round predictions to the closest integer
    return (preds == y).sum() / len(y)

def train(model:nn.Module,dataloader:Tuple[torch.tensor,torch.tensor],criterion,optimizer:optim,device:str,epochs:int)->torch.Tensor:
    losses = []
    preds = []
    labels = []
    pbar = tqdm(dataloader, desc="TRAIN")
    for (input, label) in pbar:
        input, label = input.to(device), label.to(device)
        pred = model(input)
        loss = criterion(pred, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #acc = binary_accuracy(pred, label).item()

        losses.append(loss.item())
        preds.append(pred.detach().cpu())
        labels.append(label.detach().cpu())

        # print("Epoch: {},  Iteration: {}/{},  loss:{}".format(epoch,
        #      idx, len(train_loader), loss))
        pbar.set_postfix(loss=loss)
    avg_train_loss = sum(losses)/len(losses)
    #acc = binary_accuracy(torch.cat(preds), torch.cat(labels))
    #print('avg train loss:', avg_train_loss, "avg train acc", acc)
    #report = classification_report(torch.cat(labels), torch.cat(preds))
    #print(report)
    #return acc

def main(args) -> None:
    if (not args.no_cuda) and torch.cuda.is_available():
        device = "cuda"
        
    else:
        device = "cpu"
    print(f"Using {device}")

    mel_spectrogram = T.MelSpectrogram(
        sample_rate=args.sample_rate,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )

    train_dataset = WakeWordDataset(
        args.train_data_json, audio_transformation=mel_spectrogram, device=device)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size)

    #test_dataset = WakeWordDataset(
    #    args.test_data_json, audio_transformation=mel_spectrogram, device=device)
    #test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size)

    model_cnn = CNNNetwork().to(device)

    criterion = CrossEntropyLoss().to(device)
    optimizer = optim.AdamW(model_cnn.parameters(), lr=args.lr)

    for epoch in tqdm(range(args.epochs), desc="Epoch"):
    # train model
        train(model_cnn, train_dataloader, criterion,
            optimizer, device, epoch)

    # save model
    torch.save(model_cnn.state_dict, args.save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="""
        Script to train the Wake Word Detection and save the model
        """,
                                     formatter_class=argparse.RawTextHelpFormatter
                                     )
    parser.add_argument('--sample_rate', type=int, default=8000,
                        help='the number of samples collected per second, default at 8000')
    parser.add_argument('--epochs', type=int, default=100,
                        help='epoch size')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='size of batch')
    parser.add_argument('--lr', type=float, default=1e-3, help="learning rate")
    parser.add_argument('--save_path', type=str, default=None,
                        help='path to save the train model')
    parser.add_argument('--train_data_json', type=str, default=None, required=True,
                        help='path to train dat json file')
    #parser.add_argument('--test_data_json', type=str, default=None, required=True,
    #                    help='path to train dat json file')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')
    args = parser.parse_args()

    main(args)

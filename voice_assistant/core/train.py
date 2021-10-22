from model import TransformerBlock
from sklearn.metrics import classification_report
from torch.nn import Module
from typing import List, Tuple

import torch
import torchaudio


def binary_accuracy(preds, y):
    # round predictions to the closest integer
    rounded_preds = torch.round(preds)
    acc = rounded_preds.eq(y.view_as(rounded_preds)).sum().item() / len(y)
    return acc


def save_checkpoint(checkpoint_path: str, model: Module, optimizer: torch.optim, scheduler, model_params, epoch: str = None):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, checkpoint_path)


@torch.no_grad()
def test(test_loader: List[Tuple[torch.Tensor, torch.Tensor]], model: Module, device, epoch: int):
    print(f"starting test for epoch {epoch}")
    accs = []
    preds = []
    labels = []
    for idx, (mfcc, label) in enumerate(test_loader):
        mfcc, label = mfcc.to(device), label.to(device)
        output = model(mfcc)
        pred = torch.sigmoid(output)
        acc = binary_accuracy(pred, label)

        accs.append(acc)
        preds.append(torch.flatten(torch.round(pred)).cpu())
        labels.append(torch.flatten(label).cpu())

        print("Epoch: {},  Iteration: {}/{},  accuracy: {}".format(epoch, idx,
              len(test_loader), acc, end='\n'))
        average_acc = sum(accs)/len(accs)
    print('Average test Accuracy:', average_acc, "\n")
    report = classification_report(labels, preds)
    print(report)
    return average_acc, report


def train(train_loader: List[Tuple[torch.Tensor, torch.Tensor]], model: Module, optimizer: torch.optim, loss_fct: torch.nn, device, epoch: int):
    print(f"starting train for epoch {epoch}")
    losses = []
    preds = []
    labels = []
    for idx, (mfcc, label) in enumerate(train_loader):
        mfcc, label = mfcc.to(device), label.to(device)
        optimizer.zero_grad()
        output = model(mfcc)
        loss = loss_fct(torch.flatten(output), label)
        loss.backward()
        optimizer.step()
        pred = torch.sigmoid(output)

        losses.append(loss)
        preds.append(torch.flatten(torch.round(pred).cpu()))
        labels.append(torch.flatten(label).cpu())

        print("Epoch: {},  Iteration: {}/{},  loss:{}".format(epoch,
              idx, len(train_loader), loss))
    avg_train_loss = sum(losses)/len(losses)
    acc = binary_accuracy(torch.Tensor(preds), torch.Tensor(labels))
    print('avg train loss:', avg_train_loss, "avg train acc", acc)
    report = classification_report(labels, preds)
    print(report)
    return acc, report

from ast import parse
from dataset import WakeWordData, collate_fn
from model import LSTMBinaryClassifier, TransformerBlock
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from typing import List, Tuple
from tqdm import tqdm

import argparse
import os
import torch
import torch.optim as optim
import torch.nn as nn


def binary_accuracy(preds: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # round predictions to the closest integer
    return (preds == y).sum() / len(y)


@torch.no_grad()
def test(
    test_loader: List[Tuple[torch.Tensor, torch.Tensor]],
    model: nn.Module,
    device: str,
    epoch: int,
) -> float:
    accs = []
    preds = []
    labels = []
    pbar = tqdm(test_loader, desc="TEST")
    for (mfcc, label) in pbar:
        mfcc, label = mfcc.to(device), label.to(device)
        output = model(mfcc)
        pred = torch.sigmoid(output)
        acc = binary_accuracy(pred, label)

        accs.append(acc)
        preds.append(pred.cpu())
        labels.append(label.cpu())
    average_acc = sum(accs) / len(accs)
    return average_acc


def train(
    train_loader: List[Tuple[torch.Tensor, torch.Tensor]],
    model: nn.Module,
    optimizer: torch.optim,
    loss_fct: torch.nn,
    device,
    epoch: int,
) -> float:
    losses = []
    preds = []
    labels = []
    accs = []
    pbar = tqdm(train_loader, desc="TRAIN")
    for (mfcc, label) in pbar:
        mfcc, label = mfcc.to(device), label.to(device)
        optimizer.zero_grad()
        pred = model(mfcc)
        loss = loss_fct(pred, label)
        loss.backward()
        optimizer.step()
        acc = binary_accuracy(pred, label).item()
        accs.append(acc)

        losses.append(loss.item())
        preds.append(pred.detach().cpu())
        labels.append(label.detach().cpu())

        pbar.set_postfix(loss=loss, acc=acc)
    average_acc = sum(accs) / len(accs)
    return average_acc


def main(args) -> None:
    device = torch.device(
        "cuda" if not args.no_cuda and torch.cuda.is_available() else "cpu"
    )

    """Create dataset and net"""
    train_dataset = WakeWordData(
        json_path=args.train_data_json, sample_rate=args.sample_rate, valid=False
    )
    test_dataset = WakeWordData(
        json_path=args.test_data_json, sample_rate=args.sample_rate, valid=True
    )

    kwargs = {"num_workers": args.num_workers} if not args.no_cuda else {}

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        **kwargs
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        **kwargs
    )

    "Model init"
    model_param = {
        "feature_size": 40,
        "hidden_size": args.hidden_size,
        "num_layers": 1,
        "num_classes": 1,
        "dropout": 0.1,
    }
    model_lstm = LSTMBinaryClassifier(**model_param)
    model_lstm = model_lstm.to(device)

    optimizer = optim.AdamW(
        model_lstm.parameters(), lr=args.lr, weight_decay=0.5
    )  # args.decay)
    criterion = nn.BCELoss().to(device)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )

    """Training init"""
    best_train_acc, best_train_report = 0, None
    best_test_acc, best_test_report = 0, None

    for epoch in tqdm(range(args.epochs), desc="Epoch"):
        train_acc, train_report = train(
            train_loader, model_lstm, optimizer, criterion, device, epoch
        )
        test_acc, test_report = test(test_loader, model_lstm, device, epoch)

        # record best train and test
        if train_acc > best_train_acc:
            best_train_acc = train_acc
        if test_acc > best_test_acc:
            best_test_acc = test_acc

        # saves checkpoint if metrics are better than last
        if args.save_checkpoint_path and test_acc >= best_test_acc:
            # print("found best checkpoint. saving model as",
            #      args.save_checkpoint_path)
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model_lstm.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                args.save_checkpoint_path,
            )
            best_train_report = train_report
            best_test_report = test_report

        scheduler.step(train_acc)

    # print("Done Training...")
    # print("Best Model Saved to", args.save_checkpoint_path)
    # print("\nTrain Report \n")
    # print(best_train_report)
    # print("\nTest Report\n")
    # print(best_test_report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""
        Script to train the Wake Word Detection and save the model
        """,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--sample_rate",
        type=int,
        default=8000,
        help="the number of samples collected per second, default at 8000",
    )
    parser.add_argument("--epochs", type=int, default=100, help="epoch size")
    parser.add_argument("--batch_size", type=int, default=32, help="size of batch")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument(
        "--save_checkpoint_path",
        type=str,
        default=None,
        help="path to save the train model",
    )
    parser.add_argument(
        "--train_data_json",
        type=str,
        default=None,
        required=True,
        help="path to train dat json file",
    )
    parser.add_argument(
        "--test_data_json",
        type=str,
        default=None,
        required=True,
        help="path to train dat json file",
    )
    parser.add_argument(
        "--no_cuda", action="store_true", default=False, help="disables CUDA training"
    )
    parser.add_argument(
        "--num_workers", type=int, default=1, help="number of data loading workers"
    )
    parser.add_argument(
        "--hidden_size", type=int, default=128, help="number of hidden layers"
    )
    args = parser.parse_args()

    main(args)

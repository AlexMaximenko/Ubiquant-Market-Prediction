from numpy import std
import torch
from SimpleMarketDataset import SimpleMarketDataset
from models.SimpleNet import LightningSimpleNet
from torch.utils.data import DataLoader, random_split
from pytorch_lightning import Trainer
import argparse
from pytorch_lightning.loggers import TensorBoardLogger
import logging
import sys

logger = TensorBoardLogger("/home/ivainn/Alex/Ubiquant-Market-Prediction/tb-logs", name='simple_model')

def train(args):
    GPU_NUM = args.gpus
    BATCH_SIZE = args.batch_size
    
    model = LightningSimpleNet().float()
    train_dataset = SimpleMarketDataset(args.train_data_path)
    val_dataset = SimpleMarketDataset(args.val_data_path)
    
    train_loader = DataLoader(
        dataset=train_dataset, 
        batch_size=BATCH_SIZE)
    val_loader = DataLoader(
        dataset=val_dataset, 
        batch_size=BATCH_SIZE)

    trainer = Trainer(
        gpus=GPU_NUM,
        max_epochs=10,
        strategy='ddp',
        logger=logger
    )
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data_path',  type=str, default='/home/ivainn/Alex/kaggle/data/train.csv', required=False, help='Train data in csv format')
    parser.add_argument('--val_data_path',  type=str, default='/home/ivainn/Alex/kaggle/data/val.csv', required=False, help='Val data in csv format')
    parser.add_argument('--gpus', type=int, default=2, required=False, help='Gpu num for training')
    parser.add_argument('--batch_size', type=int, default=2**14, required=False, help='Batch size for training')
    args = parser.parse_args()
    train(args)
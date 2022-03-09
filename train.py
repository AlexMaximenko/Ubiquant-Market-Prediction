from numpy import std
import torch
from data.SimpleMarketDataset import SimpleMarketDataset
from data.PerTimeDataset import PerTimeDataset
from models.SimpleNet import LightningSimpleNet
from models.MemoryNet import LightningMemoryNet
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.plugins import DDPPlugin
import argparse
from pytorch_lightning.loggers import TensorBoardLogger
import logging
import sys

logger = TensorBoardLogger("/home/ivainn/Alex/Ubiquant-Market-Prediction/tb-logs", name='simple_model')

def train_memory_model(args):
    GPU_NUM = args.gpus
    logger = TensorBoardLogger(args.tb_logs_dir, name=args.model_type)
    model = LightningMemoryNet().float()
    train_dataset = PerTimeDataset(args.train_data_path)
    val_dataset = PerTimeDataset(args.val_data_path)
    train_loader = DataLoader(
        dataset=train_dataset, 
        batch_size=1,
        shuffle=False)
    val_loader = DataLoader(
        dataset=val_dataset, 
        batch_size=1,
        shuffle=False)
    if GPU_NUM > 0:
        trainer = Trainer(max_epochs=args.epochs_num, gpus=GPU_NUM, strategy='ddp', logger=logger)
    else:
        trainer = Trainer(max_epochs=args.epochs_num, accelerator='cpu', logger=logger)
    trainer.fit(model, train_loader, val_loader)

def train_simple_model(args):
    GPU_NUM = args.gpus
    BATCH_SIZE = args.batch_size
    logger = TensorBoardLogger(args.tb_logs_dir, name=args.model_type)
    model = LightningSimpleNet().float()
    train_dataset = SimpleMarketDataset(args.train_data_path)
    val_dataset = SimpleMarketDataset(args.val_data_path)
    train_loader = DataLoader(
        dataset=train_dataset, 
        batch_size=BATCH_SIZE,
        shuffle=True)
    val_loader = DataLoader(
        dataset=val_dataset, 
        batch_size=BATCH_SIZE,
        shuffle=True)
    if GPU_NUM > 0:
        trainer = Trainer(max_epochs=args.epochs_num, gpus=GPU_NUM, strategy='ddp', logger=logger)
    else:
        trainer = Trainer(max_epochs=args.epochs_num, accelerator='cpu', logger=logger)
    trainer.fit(model, train_loader, val_loader)


def train(args):
    if args.model_type == 'memory_model':
        train_memory_model(args)
    elif args.model_type == 'simple_model':
        train_simple_model(args)


def configure_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data_path',  type=str, default='/home/ivainn/Alex/kaggle/data/train.pkl', required=False, help='Train data in pkl format')
    parser.add_argument('--val_data_path',  type=str, default='/home/ivainn/Alex/kaggle/data/val.pkl', required=False, help='Val data in pkl format')
    parser.add_argument('--gpus', type=int, default=2, required=False, help='Gpu num for training')
    parser.add_argument('--batch_size', type=int, default=2**14, required=False, help='Batch size for training')
    parser.add_argument('--model_type', type=str, default='memory_model', required=False, help='Model for training: memory_model or simple_model')
    parser.add_argument('--tb_logs_dir', type=str, default='/home/ivainn/Alex/Ubiquant-Market-Prediction/tb-logs', required=False, help='Directory for tensorboard logs')
    parser.add_argument('--epochs_num', type=int, default=10, required=False, help='Num of training epochs')
    return parser

if __name__ == "__main__":
    parser = configure_parser()
    args = parser.parse_args()
    train(args)
import pandas as pd
import argparse
import os

def split_dataset(args):
    """
    Split full dataset into val and train
        args:
            args.path_to_csv:  Path to full dataset csv
            args.output_dir: Output dir for val.csv and train.csv
            args. 
    """
    data = pd.read_csv(args.path_to_csv)
    train = data[0:int(len(data)*args.valid_percent)]
    val = data[int(len(data)*args.valid_percent):]
    train.to_csv(os.path.join(args.output_dir, 'train.csv'))
    val.to_csv(os.path.join(args.output_dir, 'val.csv'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--path_to_csv', 
        type=str,
        default='/home/ivainn/Alex/kaggle/kaggle_input/train.csv',
        help='Path to train.csv')
    parser.add_argument(
        '--output_dir',
        type=str,
        default='/home/ivainn/Alex/kaggle/data',
        help='Path to output dir for val.csv and train.csv'
    )
    parser.add_argument(
        '--valid_percent',
        type=float,
        default=0.8,
        help='Valid Percent'
    )
    args = parser.parse_args()
    split_dataset(args)
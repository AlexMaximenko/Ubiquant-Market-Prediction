import pandas as pd
import argparse
import os

def split_dataset(args):
    """
    Split full dataset into val and train
        args:
            args.path_to_data:  Path to full dataset pkl
            args.output_dir: Output dir for val.pkl and train.pkl
    """
    data = pd.read_parquet(args.path_to_parquet)
    train = data[0:int(len(data)*args.train_percent)]
    val = data[int(len(data)*args.train_percent):]
    train.to_pickle(os.path.join(args.output_dir, 'ubiquant_train.pickle'))
    val.to_pickle(os.path.join(args.output_dir, 'ubiquant_val.pickle'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--path_to_parquet', 
        type=str,
        default='/home/ubuntu/data/train_small.parquet',
        help='Path to train.parquet')
    parser.add_argument(
        '--output_dir',
        type=str,
        default='/home/ubuntu/data/ubiquant',
        help='Path to output dir for val.pickle and train.pickle'
    )
    parser.add_argument(
        '--train_percent',
        type=float,
        default=0.8,
        help='Valid Percent'
    )
    args = parser.parse_args()
    split_dataset(args)
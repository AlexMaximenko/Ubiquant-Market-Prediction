import pandas as pd
import argparse
import os

def split_dataset(args):
    """
    Split full dataset into val and train
        args:
            args.path_to_csv:  Path to full dataset pkl
            args.output_dir: Output dir for val.pkl and train.pkl
    """
    data = pd.read_pickle(args.path_to_pkl)
    train = data[0:int(len(data)*args.valid_percent)]
    val = data[int(len(data)*args.valid_percent):]
    train.to_pickle(os.path.join(args.output_dir, 'train.pkl'))
    val.to_pickle(os.path.join(args.output_dir, 'val.pkl'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--path_to_pkl', 
        type=str,
        default='/home/ivainn/Alex/kaggle/kaggle_input/prepared_train.pkl',
        help='Path to train.pkl')
    parser.add_argument(
        '--output_dir',
        type=str,
        default='/home/ivainn/Alex/kaggle/data',
        help='Path to output dir for val.pkl and train.pkl'
    )
    parser.add_argument(
        '--valid_percent',
        type=float,
        default=0.8,
        help='Valid Percent'
    )
    args = parser.parse_args()
    split_dataset(args)
from glob import glob
import numpy as np
import pandas as pd
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--path', type=str, default='./data/train/')
    arg('--output', type=str, default='./train.csv')
    args = parser.parse_args()

    files = glob(f"{args.path}/**/*.*", recursive=True)
    images = [file for file in files if file.endswith('.png')]
    masks = [file for file in files if file.endswith('.npy')]

    df = pd.DataFrame()
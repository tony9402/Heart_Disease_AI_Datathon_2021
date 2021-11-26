from glob import glob
import pandas as pd
import os
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--path', type=str, default='./data')
    arg('--save', type=str, default='./data.csv')
    args = parser.parse_args()

    df = pd.DataFrame()
    files = glob(os.path.join(args.path, '*/**/*.png'), recursive=True)

    for file in sorted(files, key=lambda x:x.split('/')[-1].split('.')[0]):
        path = os.path.abspath(file)
        path_before_extension = path.split('.')[0]
        image, mask = f"{path_before_extension}.png", f"{path_before_extension}.npy"
        train_val, clas = path.split('/')[-3:-1]
        df = df.append([[image, mask, clas, train_val]])

    df.to_csv(args.save, header=False, index=False)
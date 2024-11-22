import pandas as pd


def load_melons():
    df = pd.read_csv('./dataset/data.txt', sep=',')
    return df


def load_tests():
    df = pd.read_csv('./dataset/test.txt', sep=',')
    return df

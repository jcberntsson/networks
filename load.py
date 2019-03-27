import csv
import pandas as pd
from sklearn import svm

train = pd.read_csv('heart.csv', header=0)

if __name__ == "__main__":
    print(train.describe())
    print(train["age"].median())

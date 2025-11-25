#aici o sa inceapa cel mai frumix model


import numpy as np
import pickle
import pandas as pd
import time

np.random.seed(74)
print("✓ Imports complete")


train_file = "../input/extended_mnist_train.pkl"
test_file = "../input/extended_mnist_test.pkl"

with open(train_file, "rb") as fp:
    train = pickle.load(fp)

with open(test_file, "rb") as fp:
    test = pickle.load(fp)
import pandas as pd
from Model import *
import sys
import time

""" from collections import Counter
    num_of_key = Counter()
    for k in data['Unnamed: 27']:
        num_of_key[k] += 1
    print(num_of_key.most_common(5))

"""

start_time = time.time()

def main(data_file,split_pc):
    data = pd.read_csv(data_file)
    data = preprocess(data)
    data = encode(data)
    X_train, Y_train, X_test, Y_test = split_data(data, split_pc)
    X_train, Y_train = balance_data(X_train, Y_train)
    acc,conf = classify_svm(X_train, Y_train, X_test, Y_test)
    print(f'the accuracy for this model is {acc} and the confusion matrix for our model is {conf}')
    print("--- %s seconds ---" % (time.time() - start_time))

if __name__ == "__main__":
    if len(sys.argv) > 1:
        data_file = str(sys.argv[1])
    else:
        data_file = '.\data\DataScienceChallenge.csv'
    split_pc = 0.8
    main(data_file,split_pc)





import pandas as pd
from Model import *
import sys
import time


""" 
This docstring is for validation of the encoded values.
    from collections import Counter
    num_of_key = Counter()
    for k in data['Unnamed: 27']:
        num_of_key[k] += 1
    print(num_of_key.most_common(3))

"""

start_time = time.time()

def main(data_file,split_pc,test_file):
    data = pd.read_csv(data_file)
    data = preprocess(data)
    data = encode(data)
    X_train, Y_train, X_test, Y_test = split_data(data, split_pc)
    X_train, Y_train = balance_data(X_train, Y_train)
    acc, conf = classify_svm(X_train, Y_train, X_test, Y_test)
    print(f'the accuracy for this model using 20% of the data_file as testset is {acc} and the confusion matrix for our model is {conf}')
    print("--- %s seconds ---" % (time.time() - start_time))
    if test_file != None: #assuming the testset is clean and the last column is the target value
        test = pd.read_csv(test_file)
        X_test = encode(test)
        Y_test = X_test.pop(X_test.columns[-1])
        acc, conf = classify_svm(X_train, Y_train, X_test, Y_test)
        print(f'the accuracy for this model using test_file is {acc} and the confusion matrix for our model is {conf}')
        print("--- %s seconds ---" % (time.time() - start_time))



if __name__ == "__main__":
    if len(sys.argv) > 1:
        test_file = str(sys.argv[1])
    else:
        test_file = None
    data_file = '.\data\DataScienceChallenge.csv'
    split_pc = 0.8
    main(data_file,split_pc,test_file)





from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
import numpy as np
from sklearn import metrics

column = None


def preprocess(df):
    """
     delete empty column and clean the data
    :param df: raw pandas dataframe
    :return: prepared dataset for the next encoding step
    """
    global column
    column = df.columns
    df = df.drop(labels=column[-1], axis=1)  # I know that last column is empty, so I just removed it
    df.loc[df['Unnamed: 27'] == '2'] = np.nan # I also checked a number 2 instead of NAN or Fluent in the label column which I fixed
    column = column[:-1]
    return df


def encode(df):
    """
    encode categorical and numberical dataset
    :param df: preprocess df
    :return: encoded dataset
    """
    from collections import defaultdict
    encoder = defaultdict(LabelEncoder)
    df = df.apply(lambda x: encoder[x.name].fit_transform(x.astype(str)))
    return df

def split_data(df, split_pct):
    """
    splitting dataframe into train and
    :param df: encoded and preprocessed csv dataset
    :param split_pct: training set percentage
    :return: training data, test data and associated labels
    """
    df_copy = df.copy()
    X_train= df_copy.sample(frac=split_pct, random_state=0)
    X_test = df_copy.drop(X_train.index)
    Y_train = X_train.pop(column[-1])
    Y_test = X_test.pop(column[-1])
    print('Data has been succesfully split')

    return X_train, Y_train, X_test, Y_test

def balance_data(X_train, Y_train):
    sm = SMOTE(random_state=2)
    X_train, Y_train= sm.fit_sample(X_train, Y_train.ravel())
    print('data has been balanced for the less populated class which in our case is "Fluent"')

    return X_train, Y_train


def classify_svm(X_train,Y_train,X_test,Y_test):
    """
    We build a classifier using oversampled data X_train
    :param X_train: training data
    :param Y_train: training label
    :param X_test: testing data
    :param Y_test: testing label
    :return: acc of the classifier
    """
    """
    ### We use Grid search to find the best value for parameter C and type of kernel. ###
    from sklearn.model_selection import GridSearchCV
    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                         'C': [0.1, 1, 10, 100, 1000]},
                        {'kernel': ['linear'], 'C': [0.1, 1, 10, 100]}]

    clf = GridSearchCV(SVC(), tuned_parameters, cv=5, scoring='accuracy')
    clf.fit(X_train, Y_train)
    best_par = clf.best_params_
    print(best_par)
    >>>{'C': 0.1, 'kernel': 'linear'}
    """
    print('training....')
    clf = SVC(C=0.1, kernel='linear')
    clf.fit(X_train, Y_train)
    print('testing...')
    pred = clf.predict(X_test)
    acc = accuracy_score(Y_test, pred)
    conf = metrics.confusion_matrix(Y_test, pred)

    return acc , conf



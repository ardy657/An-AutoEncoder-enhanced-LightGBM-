from keras.models import load_model
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.metrics import accuracy_score,roc_auc_score,roc_curve,recall_score,confusion_matrix,f1_score,precision_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE  #
from numpy.random import seed
seed(1)

def read_file(dataname):
    # read data
    d = pd.read_csv(dataname)
    # Delete the time column and standardize the amount column
    #data = d.drop(['Time'], axis=1)
    d['Hour'] = d["Time"].apply(lambda x: divmod(x, 3600)[0])
    d[['Amount', 'Hour']] = StandardScaler().fit_transform(d[['Amount', 'Hour']])
    #d['Amount'] = StandardScaler().fit_transform(d[['Amount']])
    d.drop('Time', axis=1, inplace=True)
    # Extract the negative sample (normal data; label: 0), and cut it into training set test set according to the proportion of 8:2.
    mask = (d['Class'] == 0)
    X_train, X_test = train_test_split(d[mask], test_size=0.3, random_state=0)
    X_train = X_train.drop(['Class'], axis=1).values
    X_test = X_test.drop(['Class'], axis=1).values

    # Extract all positive samples (exception data, label: 1) as part of the test set
    X_fraud = d[~mask].drop(['Class'], axis=1).values

    return X_train, X_test, X_fraud
if __name__ == '__main__':
    dataname = (r'D:\00 master working\program\Anomaly Detection\data\creditcardfraud\creditcard.csv')
    X_train, X_test, X_fraud = read_file(dataname)
    print(" X_train:")
    print(X_train)
    print("X_test：")
    print(X_test)
    print("X_fraud:")
    print(X_fraud)
    # Read model
    autoencoder = load_model('kaggle2_relu_model_50epoch_30_16_8.h5')
    # Use the trained autoencoder to reconstruct the test set (X_test; X_fraud: the former is normal sample, and the latter is all abnormal samples)
    pred_test = autoencoder.predict(X_test)
    pred_fraud = autoencoder.predict(X_fraud)
    ## Use the trained autoencoder to reconstruct GBDT training set and test set (X_test; X_fraud: the former is normal sample, and the latter is all abnormal samples)
    mse_test = np.mean(np.power(X_test - pred_test, 2), axis=1)
    mse_fraud = np.mean(np.power(X_fraud - pred_fraud, 2), axis=1)

    mae_test = np.mean(np.abs(X_test - pred_test), axis=1)
    mae_fraud = np.mean(np.abs(X_fraud - pred_fraud), axis=1)
    mse_df = pd.DataFrame()
    mse_df['Class'] = [0] * len(mse_test) + [1] * len(mse_fraud)
    mse_df['MSE'] = np.hstack([mse_test, mse_fraud])
    mse_df['MAE'] = np.hstack([mae_test, mae_fraud])
    mse_df = mse_df.sample(frac=1).reset_index(drop=True)
    mse_df.to_csv('2.2_mse_df.csv')

    thre= mse_df['MAE'].groupby(mse_df['Class']).describe()
    print('threshold',thre)
    thre.to_csv('2.2_thre.csv')
    threshold1 = thre['mean'][0]
    threshold2 = thre['mean'][1]
    threshold=0.5
    print('threshold',threshold)


    mse_df.loc[mse_df['MAE'] >= threshold, 'class_pre'] = 1
    mse_df.loc[mse_df['MAE'] < threshold, 'class_pre'] = 0

    label_real = np.array(mse_df['Class'])
    class_pre = np.array(mse_df['class_pre'])

    accuracy_AE = accuracy_score(label_real, class_pre)
    print('The accuracy_AE is', accuracy_AE)

    prec_score = precision_score(label_real, class_pre)
    print('prec_score', prec_score)

    reca_score = recall_score(label_real, class_pre)
    print('reca_score', reca_score)
    # F1score
    f1_score = f1_score(label_real, class_pre)
    print('f1_score', f1_score)
    # con_mirx
    con_mirx = confusion_matrix(label_real, class_pre, labels=[1, 0])
    print('con_mirx', con_mirx)


    # G-mean
    G_mean = np.sqrt(
        (con_mirx[0, 0] / (con_mirx[0, 0] + con_mirx[1, 0])) * (con_mirx[0, 1] / (con_mirx[0, 1] + con_mirx[1, 1])))
    print('G_mean', G_mean)
    # AUC
    auc_AE = roc_auc_score(label_real, class_pre)
    print('The auc_AE is', auc_AE)
    #MCC
    MCC = (con_mirx[0, 0] * con_mirx[1, 1] - con_mirx[1, 0] * con_mirx[0, 1]) / np.sqrt((con_mirx[0, 0] + con_mirx[1, 0]) * (con_mirx[0, 0] + con_mirx[0, 1]) * (con_mirx[1, 1] + con_mirx[1, 0]) * (con_mirx[1, 1] + con_mirx[0, 1]))
    print('The MCC is', MCC)

    """

    The sample() parameter frac is the proportion to be returned. For example, if there are 10 rows of data in df, I only want to return 30% of them, so frac=0.3
    set_ Index() and reset_ The difference between oindex() and the former is that the existing dataframe setting is different from the previous index; The latter is the restore and initial index method: 0,1,2,3,4

    """
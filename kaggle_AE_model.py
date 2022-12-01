import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Dense
from keras.callbacks import ModelCheckpoint
from keras import regularizers
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from numpy.random import seed
seed(1)


col_n=['V1','V2','V3','V4','V5','V6','V7','V8','V9','V10','V11','V12','V13',
       'V14','V15','V16','V17','V18','V19','V20','V21','V22','V23','V24','V25','V26','V27','V28','Amount','Hour']
def read_file(dataname):
    # read data
    d = pd.read_csv(dataname)

    num_nonfraud = np.sum(d['Class'] == 0)  # Class 0 sample represents non fraud sample, which is normal sample data. There are 284315 pieces of data in this data set, accounting for a large proportion
    num_fraud = np.sum(d['Class'] == 1)  # Class 1 samples refer to fraud samples, which are abnormal samples in anomaly detection. There are only 492 pieces of data in this data set, a small proportion.
    plt.bar(['fraud', 'non_fraud'], [num_fraud, num_nonfraud], color='dodgerblue')
    plt.show()

    d['Hour'] = d["Time"].apply(lambda x: divmod(x, 3600)[0])
    d[['Amount', 'Hour']] = StandardScaler().fit_transform(d[['Amount', 'Hour']])
    d.drop('Time',axis=1,inplace=True)
    # Extract the negative sample (normal data; label: 0), and cut it into training set test set according to the proportion of 8:2.
    mask = (d['Class'] == 0)
    X_train, X_test = train_test_split(d[mask], test_size=0.3, random_state=0)
    print(X_train)
    print(X_test)
    X_train = X_train.drop(['Class'], axis=1).values
    X_test = X_test.drop(['Class'], axis=1).values
    X_train=pd.DataFrame(X_train,columns=col_n)
    X_test=pd.DataFrame(X_test,columns=col_n)
    print(X_train.shape)
    print(X_train)
    print(X_test.shape)
    print(X_test)


    # Extract all positive samples (exception data, label: 1) as part of the test set
    X_fraud = d[~mask].drop(['Class'], axis=1).values
    print(X_fraud.shape)
    X_fraud=pd.DataFrame(X_fraud,columns=col_n)
    print(X_fraud)
    return X_train, X_test, X_fraud


if __name__ == '__main__':

    dataname = (r'D:\00 master working\program\Anomaly Detection\data\creditcardfraud\creditcard.csv')

    X_train, X_test, X_fraud = read_file(dataname)
    input_dim = X_train.shape[1]#30#20
    print(input_dim)
    encoding_dim = 16
    num_epoch = 100#50#100
    batch_size = 32



    # keras
    input_layer = Input(shape=(input_dim,))
    encoder = Dense(encoding_dim, activation="relu",
                    activity_regularizer=regularizers.l1(10e-5))(input_layer)
    encoder = Dense(int(8), activation="relu")(encoder)
    decoder = Dense(int(8), activation="relu")(encoder)
    decoder = Dense(input_dim, activation="relu")(decoder)
    autoencoder = Model(inputs=input_layer, outputs=decoder)
    autoencoder.compile(optimizer="adam",
                        loss="mean_squared_error",
                        metrics=['mae'])

    # Save the model as sofasofa_ Model. h5 and start training.
    checkpointer = ModelCheckpoint(filepath="1.14_kaggle_relu_model_100epoch_30_16_8.h5",
                                   verbose=0,
                                   save_best_only=True)
    history = autoencoder.fit(X_train, X_train,
                              epochs=num_epoch,
                              batch_size=batch_size,
                              shuffle=True,
                              validation_data=(X_test, X_test),
                              verbose=1,
                              callbacks=[checkpointer]).history

    # Draw the loss function curve
    plt.figure(figsize=(14, 5))
    plt.subplot(121)
    plt.plot(history["loss"], c='dodgerblue', lw=3)
    plt.plot(history["val_loss"], c='coral', lw=3)
    plt.title('model loss')
    plt.ylabel('mse')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')

    plt.subplot(122)
    plt.plot(history['mae'], c='dodgerblue', lw=3)
    plt.plot(history['val_mae'], c='coral', lw=3)
    plt.title('model_mae')
    plt.ylabel('mae')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')
    plt.show()

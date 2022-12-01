
#coding=utf-8

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn import metrics
from sklearn.metrics import accuracy_score,roc_auc_score,roc_curve,recall_score,confusion_matrix,f1_score,precision_score
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.model_selection import KFold
from sklearn import preprocessing
#from imblearn.over_sampling import SMOTE
import gc
import cProfile
from memory_profiler import profile
from sklearn.utils import shuffle

import time
from functools import wraps

#Functions that measure function call time
def fn_timer(function):
    @wraps(function)
    def function_timer(*args, **kwargs):
        t0 = time.time()
        result = function(*args, **kwargs)
        t1 = time.time()
        print("Total time running %s: %s seconds" %
              (function.__name__, str(t1 - t0))
              )
        return result

    return function_timer


def creatDtype():
    dtype={#'id':'object',
         'label':'int64',
         #'date':'int64',
         'f1': 'float64',
         'f2': 'float64',
         'f3': 'float64',
         'f4': 'float64',
         'f5': 'float64',
        }
    for i in range(20,298):
        dtype['f'+str(i)]='float64'
    for i in range(6, 20):
        dtype['f' + str(i)] = 'float64'
    return dtype

# Evaluation function
def evalMetric(preds, dtrain):
    label = dtrain.get_label()
    fpr,tpr,threholds= roc_curve(label, preds, pos_label=1)
    #receiver operating characteristic(ROC)
    res = 0.0
    ths= [0.001,0.005,0.01]
    weight = [0.4,0.3,0.3]
    for wi,th in enumerate(ths):
        index = 1
        for index in range(1,len(fpr)):
            if fpr[index-1] <=th and fpr[index]>=th:
                res = res+(tpr[index-1] + tpr[index])/2 * weight[wi]
    return 'feval',res,True
#Divide the data into k copies according to the index, which can be used to generate the output of the first layer, and ensure that there is no interference between the data
def Kfolds(x_index,k=5,seed = 1):
    np.random.seed(seed)
    xL = np.array_split(np.random.choice(x_index,len(x_index),replace=False),k)
    return xL

#The indexes of the validation set and the training set are respectively expressed (that is, the index of the ith fold data in Kfolds is used as the prediction set, and the rest is used as the training set)
def GroupSelect(xL,i=0):
    xLc = xL.copy()
    pre_index = list(xLc.pop(i))
    train_index = sum([list(x) for x in xLc],[])
    return train_index,pre_index

''' 
train_x:data
train_y:label
'''
def TrainSet(x,y,xL,i=0):
    train_index,pre_index = GroupSelect(xL,i)
    train_x,pre_x = x.loc[train_index],x.loc[pre_index]
    train_y,pre_y = y.loc[train_index],y.loc[pre_index]#??????????
    return train_x,train_y,pre_x,pre_y


#K-fold cross validation algorithm for lightgbm algorithm


@fn_timer
#@profile()
def KfoldsTrain_light(x,y,test,k=5,lgb_params=None):
    if lgb_params == None:
        lgb_params = {
            'boosting_type':'goss',#'gbdt',
            'objective':'binary',
            # 'nthread':-1,
            'nthread': 4,
            'meteic':'auc',
            # 'num_leaves':7,
            'num_leaves': 2 ** 5,
            'min_child_samples':1000
        }
    xL=Kfolds(np.array(x.index),k)
    test_predict_proba = pd.DataFrame()
    for i in range(k):
        print('begin the '+str(i)+' of '+str(k)+' kflods training...')
        train_x,train_y,pre_x,pre_y = TrainSet(x,y,xL,i)
        xgtrain = lgb.Dataset(train_x,train_y)

        xgvalid = lgb.Dataset(pre_x,pre_y)
        #pre_x  Validation Set Data pre_y Validation Set label
        evals_results = {}
        bst1 = lgb.train(lgb_params,xgtrain,valid_sets=[xgtrain,xgvalid],valid_names=['train','valid'],evals_result=evals_results,
                         num_boost_round=200,early_stopping_rounds=50,verbose_eval=50,feval=evalMetric)
        #When the verification score of 100 evaluators is not improved, the training will be stopped, so the number of evaluators actually used will not reach this number. Early stop is an effective way to select the number of evaluators, rather than setting it as another super parameter to be tuned!
        #That is, the number of boosting iterations, or the number of residual trees. The parameter name is n_estimators/num_iterations/num_round/num_boost_round。
        print('The ',i,' times running...')
        print('Best ite and score',bst1.best_iteration,bst1.best_score)
        pre_ba =  pd.Series(bst1.predict(test,num_iteration=bst1.best_iteration),index=test.index)
        test_predict_proba[i] = pre_ba
        print (test_predict_proba)
    print (test_predict_proba)
    test_predict_proba_mean = test_predict_proba.mean(axis=1)
    return test_predict_proba_mean

#This function deletes the remaining unwanted feature columns
def preprocess_likeo(df,col_need):
    num = 0
    drop_cols = []
    for col in df.columns:
        if(col not in col_need):
            drop_cols.append(col)
            num +=1
    df.drop(drop_cols,axis=1,inplace=True)
    print('drop',num, "features in data set")
    return df

def evalMetric2(y_true, y_score):
    fpr, tpr, threholds = roc_curve(y_true, y_score, pos_label=1)
    res = 0.0
    ths = [0.001, 0.005, 0.01]
    weight = [0.4, 0.3, 0.3]
    for wi, th in enumerate(ths):
        index = 1
        for index in range(1, len(fpr)):
            if fpr[index - 1] <= th and fpr[index] >= th:
                res = res + (tpr[index - 1] + tpr[index]) / 2 * weight[wi]
    return res


if __name__ == '__main__':


    train_x_path= r'D:\00 master working\program\Anomaly Detection\train_x\train_x.csv'
    train_x_data = pd.read_csv(train_x_path, low_memory=False)
    train_x = pd.DataFrame(train_x_data)
    train_x.drop('index', axis=1, inplace=True)
    train_x.drop([len(train_x) - 1], inplace=True)

    train_x=shuffle(train_x)

    test_path=r'D:\00 master working\program\Anomaly Detection\test\test.csv'
    test_data = pd.read_csv(test_path, low_memory=False)
    test = pd.DataFrame(test_data)
    test.drop([len(test) - 1], inplace=True)

    test = shuffle(test)

    train_y = train_x['label']
    train_x.drop('label',axis=1,inplace=True)
    test_id = test['id'].copy()
    test_copy = test.copy()
    test.drop('id',axis=1,inplace=True)
    test.drop('index', axis=1, inplace=True)
    test.drop('label', axis=1, inplace=True)
    train_size = len(train_x)
    all_data = train_x.append(test)
    print(all_data.shape)

    train_x = all_data[:train_size]
    test = all_data[train_size:]
    print('the size of train set is '+str(train_x.shape[0])+',the size of test set is'+str(test.shape[0]))

    # Model training
    # After setting the lightgbm parameter, call the above method
    total_res = pd.DataFrame()
    total_res[['id','label_real']] = test_copy[['id','label']]

    lgb_params={
    # Core parameters
            'boosting_type': 'goss',  # Traditional gradient lifting decision tree
            'objective': 'binary',  # Objective: secondary classification
            'metric': 'auc',
            'num_leaves': 2 ** 5,  # (32) Number of leaves on a tree, 30250, default=31
            'learning_rate': 0.1,  # 0.08,0.01
            'nthread': 4,
        # The number of threads in LightGBM. For faster speed, set this to the actual number of CPU cores, not the number of threads (most CPUs use hyper threading, and each CPU core generates two threads)
        # It is normal that the task manager or any similar CPU monitoring tool may report underutilized kernels
        # For parallel learning, you should not use all CPU cores, as this will lead to poor network performance
        # Parameters used to control the model learning process
            # 'max_depth':The default value is not - 1, which can be left unchecked. Since the data set is large, overfitting is prevented for small data sets
            'lambda_l2': 10,  #
           #'subsample': 0.85,  # Randomly select part of the data without resampling, accelerate training, and process over fitting 0.8
            'seed': 2018,  # bagging times
           # 'colsample_bytree': 0.5,  # LightGBM Some features will be selected randomly in each iteration For example, if it is set to 0.8, 80% of the features will be selected before each tree training, which can be used to speed up the training

            'min_child_weight': 1.5,  # Increase the minimum leaf weight, reduce the over fitting by 300
            #'feature_fraction': 0.5,#These can be omitted during boosting to distinguish the advantages and disadvantages of bagging and boosting
           # 'bagging_fraction': 0.9,
           #  'bagging_freq' : 10,
       # 'enable_bundle':True,
           'is_unbalance': False
    }


    total_res['score']=KfoldsTrain_light(train_x,train_y,test,10,lgb_params)

    #Classification conversion
    total_res.loc[total_res['score'] >= 0.5, 'class_pre'] = 1
    total_res.loc[total_res['score'] < 0.5, 'class_pre'] = 0
    #Calculation of evaluation index
    label_real = np.array(total_res['label_real'])
    print(label_real)
    class_pre = np.array(total_res['class_pre'])
    print(class_pre)
    # accuracy
    accuracy_lgbm = accuracy_score(label_real, class_pre)

    prec_score = precision_score(label_real, class_pre)

    print ('prec_score',prec_score)
    raca_score = recall_score(label_real, class_pre)
    # F1score
    # def f1_score(y_true, y_pred, labels=None, pos_label=1, average='binary',sample_weight=None)
    #  harmonic mean,F1 = 2 * (precision * recall) / (precision + recall)
    print('raca_score',raca_score)
    f1_score = f1_score(label_real, class_pre)
    print('f1_score', f1_score)
    #con_mirx
    con_mirx = confusion_matrix(label_real, class_pre,labels=[1,0])
    #
    print('con_mirx', con_mirx)
    #G-mean
    G_mean = np.sqrt((con_mirx[0,0]/(con_mirx[0,0]+con_mirx[1,0]))*(con_mirx[0,1]/(con_mirx[0,1]+con_mirx[1,1])))
    print('G_mean',G_mean)
    #AUC
    auc_lgbm = roc_auc_score(label_real, class_pre)

    feval_score = evalMetric2(label_real, total_res['score'])
    #MCC
    TP = con_mirx[0,0]
    FP = con_mirx[0,1]
    FN = con_mirx[1,0]
    TN = con_mirx[1,1]
    try:
        if np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)) == 0:
            MCC = 0
        else:
            MCC = (TP * TN - FP * FN) / np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    except:
        MCC = 0
    print('The feval_score is', feval_score)
    total_res['feval_score'] = feval_score

    total_res['accuracy_lgbm'] = accuracy_lgbm
    total_res['auc_lgbm'] = auc_lgbm
    print('The accuracy_lgbm is',accuracy_lgbm)
    print('The auc_lgbm is', auc_lgbm)
    print('The MCC is',MCC)

    #total_res.to_csv('submission_b_24_2.csv',index=False)
    del test_copy
    del test
    del train_x
   # del csv_df2
    gc.collect()



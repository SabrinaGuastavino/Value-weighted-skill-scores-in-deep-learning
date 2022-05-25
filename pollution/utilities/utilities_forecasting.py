#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 16:38:30 2022

@author: sabry
"""
import numpy as np
from sklearn.metrics import confusion_matrix
import datetime as dt

from keras.layers import Dense, Flatten
from keras.layers.recurrent import LSTM
from keras.models import Sequential, load_model
from keras.optimizers import Adam

from keras.models import Model
from keras.models import Input


#***
def compute_cm_tss_threshold(y_true, pred,threshold):
    # Compute confusion matrix and TSS given y_true and y_pred in categorical form
    pred_threshold = pred > threshold
    cm = confusion_matrix(y_true,pred_threshold)
    if cm.shape[0] == 1 and sum(y_true) == 0:
        a = 0.
        d = float(cm[0, 0])
        b = 0.
        c = 0.
    elif cm.shape[0] == 1 and sum(y_true) == y_true.shape[0]:
        a = float(cm[0, 0])
        d = 0.
        b = 0.
        c = 0.
    elif cm.shape[0] == 2:
        a = float(cm[1, 1])
        d = float(cm[0, 0])
        b = float(cm[0, 1])
        c = float(cm[1, 0])
    TP = a
    TN = d
    FP = b
    FN = c

    if TP + FN == 0.:
        if TP == 0.:
            tss_aux1 = 0.  # float('NaN')
        else:
            tss_aux1 = -100  # float('Inf')
    else:
        tss_aux1 = (TP / (TP + FN))

    if (FP + TN) == 0.:
        if FP == 0.:
            tss_aux2 = 0.  # float('NaN')
        else:
            tss_aux2 = -100  # float('Inf')
    else:
        tss_aux2 = (FP / (FP + TN))

    tss = tss_aux1 - tss_aux2
    
    if ((TP + FN) * (FN + TN) + (TP + FP) * (FP + TN)) == 0.:
        if (TP * TN - FN * FP) == 0:
            hss = 0.  # float('NaN')
        else:
            hss = -100  # float('Inf')
    else:
        hss = 2 * (TP * TN - FN * FP) / ((TP + FN) *
                                         (FN + TN) + (TP + FP) * (FP + TN))
    
    if TP+FP+FN==0:
        CSI = 0
    else:
        CSI = TP/(TP+FP+FN)

    
    return cm, tss, hss, CSI
        

#***
def compute_cm_tss(y_true, pred):
    # Compute confusion matrix and TSS given y_true and y_pred in categorical form
    cm = confusion_matrix(y_true,pred)
    if cm.shape[0] == 1 and sum(y_true) == 0:
        a = 0.
        d = float(cm[0, 0])
        b = 0.
        c = 0.
    elif cm.shape[0] == 1 and sum(y_true) == y_true.shape[0]:
        a = float(cm[0, 0])
        d = 0.
        b = 0.
        c = 0.
    elif cm.shape[0] == 2:
        a = float(cm[1, 1])
        d = float(cm[0, 0])
        b = float(cm[0, 1])
        c = float(cm[1, 0])
    TP = a
    TN = d
    FP = b
    FN = c

    if TP + FN == 0.:
        if TP == 0.:
            tss_aux1 = 0.  # float('NaN')
        else:
            tss_aux1 = -100  # float('Inf')
    else:
        tss_aux1 = (TP / (TP + FN))

    if (FP + TN) == 0.:
        if FP == 0.:
            tss_aux2 = 0.  # float('NaN')
        else:
            tss_aux2 = -100  # float('Inf')
    else:
        tss_aux2 = (FP / (FP + TN))

    tss = tss_aux1 - tss_aux2
    
    if ((TP + FN) * (FN + TN) + (TP + FP) * (FP + TN)) == 0.:
        if (TP * TN - FN * FP) == 0:
            hss = 0.  # float('NaN')
        else:
            hss = -100  # float('Inf')
    else:
        hss = 2 * (TP * TN - FN * FP) / ((TP + FN) *
                                         (FN + TN) + (TP + FP) * (FP + TN))
    
    if TP+FP+FN==0:
        CSI = 0
    else:
        CSI = TP/(TP+FP+FN)

    
    return cm, tss, hss, CSI

def compute_weight_cm_tss(y, pred):
    # Compute confusion matrix and TSS given y_true and y_pred in categorical form

    TN,FP,FN,TP = weighted_confusion_matrix(y,pred)

    if TP + FN == 0.:
        if TP == 0.:
            tss_aux1 = 0.  # float('NaN')
        else:
            tss_aux1 = -100  # float('Inf')
    else:
        tss_aux1 = (TP / (TP + FN))

    if (FP + TN) == 0.:
        if FP == 0.:
            tss_aux2 = 0.  # float('NaN')
        else:
            tss_aux2 = -100  # float('Inf')
    else:
        tss_aux2 = (FP / (FP + TN))

    tss = tss_aux1 - tss_aux2
    
    if ((TP + FN) * (FN + TN) + (TP + FP) * (FP + TN)) == 0.:
        if (TP * TN - FN * FP) == 0:
            hss = 0.  # float('NaN')
        else:
            hss = -100  # float('Inf')
    else:
        hss = 2 * (TP * TN - FN * FP) / ((TP + FN) *
                                         (FN + TN) + (TP + FP) * (FP + TN))
    
    if TP+FP+FN==0:
        CSI=0
    else:
        CSI = TP/(TP+FP+FN)
    
    weighted_cm = np.zeros((2,2))
    weighted_cm[0,0]=TN
    weighted_cm[0,1]=FP
    weighted_cm[1,0]=FN
    weighted_cm[1,1]=TP
    
    return weighted_cm, tss, hss, CSI

#***
def optimize_threshold_skill_scores(probability_prediction, Y_training):
    n_samples = 100
    step = 1. / n_samples
    

    xss_threshold = 0
    Y_best_predicted = np.zeros((Y_training.shape))
    tss_vector = np.zeros(n_samples)
    hss_vector = np.zeros(n_samples)
    xss_threshold_vector = np.zeros(n_samples)
    a = probability_prediction.max()
    b = probability_prediction.min()
    for threshold in range(1, n_samples):
        xss_threshold = step * threshold * np.abs(a - b) + b
        xss_threshold_vector[threshold] = xss_threshold
        Y_predicted = probability_prediction > xss_threshold
        res = metrics_classification(Y_training > 0, Y_predicted, print_skills=False)
        tss_vector[threshold] = res['tss']
        hss_vector[threshold] = res['hss']
    max_tss=np.max(tss_vector)
    max_hss=np.max(hss_vector)
    eps=1e-5
    if max_tss==0:
        max_tss=max_tss+eps
    if max_hss==0:
        max_hss=max_hss+eps
    nss_vector = 0.5*((tss_vector/max_tss) + (hss_vector/max_hss))        
    idx_best_nss = np.where(nss_vector==np.max(nss_vector))  
    print('idx best nss=',idx_best_nss)
    #best NSS
    best_xss_threshold = xss_threshold_vector[idx_best_nss]
    if len(best_xss_threshold)>1:
        best_xss_threshold = best_xss_threshold[0]
        Y_best_predicted = probability_prediction > best_xss_threshold
    else:
        Y_best_predicted = probability_prediction > best_xss_threshold
    print('best NSS')
    metrics_training = metrics_classification(Y_training > 0, Y_best_predicted)
    
    #best TSS
    idx_best_tss = np.where(tss_vector==np.max(tss_vector))  
    print('idx best tss=',idx_best_tss)
    best_xss_threshold_tss = xss_threshold_vector[idx_best_tss]
    if len(best_xss_threshold_tss)>1:
        best_xss_threshold_tss = best_xss_threshold_tss[0]
        Y_best_predicted_tss = probability_prediction > best_xss_threshold_tss
    else:
        Y_best_predicted_tss = probability_prediction > best_xss_threshold_tss
    print('best TSS')
    metrics_training_tss = metrics_classification(Y_training > 0, Y_best_predicted_tss)
    
    #best HSS
    idx_best_hss = np.where(hss_vector==np.max(hss_vector))  
    print('idx best hss=',idx_best_hss)
    best_xss_threshold_hss = xss_threshold_vector[idx_best_hss]
    if len(best_xss_threshold_hss)>1:
        best_xss_threshold_hss = best_xss_threshold_hss[0]
        Y_best_predicted_hss = probability_prediction > best_xss_threshold_hss
    else:
        Y_best_predicted_hss = probability_prediction > best_xss_threshold_hss
    print('best HSS')
    metrics_training_hss = metrics_classification(Y_training > 0, Y_best_predicted_hss)
    
    #best (TSS+HSS)/2
    comb_tss_hss = (hss_vector+tss_vector)/2.
    idx_best_tss_hss = np.where(comb_tss_hss==np.max(comb_tss_hss)) 
    print('idx best (tss+hss)/2 =',idx_best_tss_hss)
    best_xss_threshold_tss_hss = xss_threshold_vector[idx_best_tss_hss]
    if len(best_xss_threshold_tss_hss)>1:
        best_xss_threshold_tss_hss = best_xss_threshold_tss_hss[0]
        Y_best_predicted_tss_hss = probability_prediction > best_xss_threshold_tss_hss
    else:
        Y_best_predicted_tss_hss = probability_prediction > best_xss_threshold_tss_hss
    print('best (TSS+HSS)/2')
    metrics_training_tss_hss = metrics_classification(Y_training > 0, Y_best_predicted_tss_hss)
    

    return best_xss_threshold, metrics_training, nss_vector, best_xss_threshold_tss, best_xss_threshold_hss, best_xss_threshold_tss_hss, np.max(comb_tss_hss)

def metrics_classification(y_real, y_pred, print_skills=True):

    cm, far, pod, acc, hss, tss, fnfp, csi = classification_skills(y_real, y_pred)

    if print_skills:
        print ('confusion matrix')
        print (cm)
        print ('false alarm ratio       \t', far)
        print ('probability of detection\t', pod)
        print ('accuracy                \t', acc)
        print ('hss                     \t', hss)
        print ('tss                     \t', tss)
        print ('balance                 \t', fnfp)
        print ('csi                 \t', csi)

    balance_label = float(sum(y_real)) / y_real.shape[0]

    #cm, far, pod, acc, hss, tss, fnfp = classification_skills(y_real, y_pred)

    return {
        "cm": cm,
        "far": far,
        "pod": pod,
        "acc": acc,
        "hss": hss,
        "tss": tss,
        "fnfp": fnfp,
        "balance label": balance_label,
        "csi": csi}

def classification_skills(y_real, y_pred):

    cm = confusion_matrix(y_real, y_pred)

    if cm.shape[0] == 1 and sum(y_real) == 0:
        a = 0.
        d = float(cm[0, 0])
        b = 0.
        c = 0.
    elif cm.shape[0] == 1 and sum(y_real) == y_real.shape[0]:
        a = float(cm[0, 0])
        d = 0.
        b = 0.
        c = 0.
    elif cm.shape[0] == 2:
        a = float(cm[1, 1])
        d = float(cm[0, 0])
        b = float(cm[0, 1])
        c = float(cm[1, 0])
    TP = a
    TN = d
    FP = b
    FN = c

    if (TP + FP + FN + TN) == 0.:
        if (TP + TN) == 0.:
            acc = 0.  # float('NaN')
        else:
            acc = -100  # float('Inf')
    else:
        acc = (TP + TN) / (TP + FP + FN + TN)

    if TP + FN == 0.:
        if TP == 0.:
            tss_aux1 = 0.  # float('NaN')
        else:
            tss_aux1 = -100  # float('Inf')
    else:
        tss_aux1 = (TP / (TP + FN))

    if (FP + TN) == 0.:
        if FP == 0.:
            tss_aux2 = 0.  # float('NaN')
        else:
            tss_aux2 = -100  # float('Inf')
    else:
        tss_aux2 = (FP / (FP + TN))

    tss = tss_aux1 - tss_aux2

    if ((TP + FN) * (FN + TN) + (TP + FP) * (FP + TN)) == 0.:
        if (TP * TN - FN * FP) == 0:
            hss = 0.  # float('NaN')
        else:
            hss = -100  # float('Inf')
    else:
        hss = 2 * (TP * TN - FN * FP) / ((TP + FN) *
                                         (FN + TN) + (TP + FP) * (FP + TN))

    if FP == 0.:
        if FN == 0.:
            fnfp = 0.  # float('NaN')
        else:
            fnfp = -100  # float('Inf')
    else:
        fnfp = FN / FP

    if (TP + FN) == 0.:
        if TP == 0.:
            pod = 0  # float('NaN')
        else:
            pod = -100  # float('Inf')
    else:
        pod = TP / (TP + FN)


    if (TP + FP) == 0.:
        if FP == 0.:
            far = 0.  # float('NaN')
        else:
            far = -100  # float('Inf')
    else:
        far = FP / (TP + FP)

    #acc = (a + d) / (a + b + c + d)
    #tpr = a / (a + b)
    #tnr = d / (d + c)
    #wtpr = a / (a + b) * (a + c) / (a + b + c + d) + d / (c + d) * (b + d) / (a + b + c + d)
    #pacc = a / (a + c)
    #nacc = d / (b + d)
    #wacc = a / (a + c) * (a + c) / (a + b + c + d) + d / (b + d) * (b + d) / (a + b + c + d)

    # if the cm has a row or a column equal to 0, we have bad tss
    if TP+FN == 0 or TN+FP == 0 or TP+FP == 0 or TN+FN == 0:
        tss = 0
    if TP+FP+FN==0:
        csi = 0
    else:
        csi = TP/(TP+FP+FN)

    #, pod, acc, hss, tss, fnfp, tpr, tnr, pacc, nacc, wacc, wtpr)
    return cm.tolist(), far, pod, acc, hss, tss, fnfp, csi


#***
def optimize_threshold_skill_scores_weight_matrix(probability_prediction, Y_training):
    n_samples = 100
    step = 1. / n_samples

    xss_threshold = 0
    Y_best_predicted = np.zeros((Y_training.shape))
    tss_vector = np.zeros(n_samples)
    hss_vector = np.zeros(n_samples)
    xss_threshold_vector = np.zeros(n_samples)
    a = probability_prediction.max()
    b = probability_prediction.min()
    print('A:',a)
    print('B:',b)
    for threshold in range(1, n_samples):
        xss_threshold = step * threshold * np.abs(a - b) + b
        xss_threshold_vector[threshold] = xss_threshold
        Y_predicted = probability_prediction > xss_threshold
        res = metrics_classification_weight(Y_training > 0, Y_predicted, print_skills=False)
        tss_vector[threshold] = res['tss']
        hss_vector[threshold] = res['hss']

    max_tss=np.max(tss_vector)
    max_hss=np.max(hss_vector)
    eps=1e-5
    if max_tss==0:
        max_tss=max_tss+eps
    if max_hss==0:
        max_hss=max_hss+eps
    print('MAX TSS:',max_tss)
    print('MAX HSS:',max_hss)
    nss_vector = 0.5*((tss_vector/max_tss) + (hss_vector/max_hss))       
    idx_best_nss = np.where(nss_vector==np.max(nss_vector))  
    print('idx best nss=',idx_best_nss)
    #best NSS
    best_xss_threshold = xss_threshold_vector[idx_best_nss]
    if len(best_xss_threshold)>1:
        best_xss_threshold = best_xss_threshold[0]
        Y_best_predicted = probability_prediction > best_xss_threshold
    else:
        Y_best_predicted = probability_prediction > best_xss_threshold
    print('best NSS')
    metrics_training = metrics_classification_weight(Y_training > 0, Y_best_predicted)
    
    #best TSS
    idx_best_tss = np.where(tss_vector==np.max(tss_vector))  
    print('idx best tss=',idx_best_tss)
    best_xss_threshold_tss = xss_threshold_vector[idx_best_tss]
    if len(best_xss_threshold_tss)>1:
        best_xss_threshold_tss = best_xss_threshold_tss[0]
        Y_best_predicted_tss = probability_prediction > best_xss_threshold_tss
    else:
        Y_best_predicted_tss = probability_prediction > best_xss_threshold_tss
    print('best TSS')
    metrics_training_tss = metrics_classification_weight(Y_training > 0, Y_best_predicted_tss)
    
    #best HSS
    idx_best_hss = np.where(hss_vector==np.max(hss_vector))  
    print('idx best hss=',idx_best_hss)
    best_xss_threshold_hss = xss_threshold_vector[idx_best_hss]
    if len(best_xss_threshold_hss)>1:
        best_xss_threshold_hss = best_xss_threshold_hss[0]
        Y_best_predicted_hss = probability_prediction > best_xss_threshold_hss
    else:
        Y_best_predicted_hss = probability_prediction > best_xss_threshold_hss
    print('best HSS')
    metrics_training_hss = metrics_classification_weight(Y_training > 0, Y_best_predicted_hss)
    
    #best (TSS+HSS)/2
    comb_tss_hss = (hss_vector+tss_vector)/2.
    idx_best_tss_hss = np.where(comb_tss_hss==np.max(comb_tss_hss)) 
    print('idx best (tss+hss)/2 =',idx_best_tss_hss)
    best_xss_threshold_tss_hss = xss_threshold_vector[idx_best_tss_hss]
    if len(best_xss_threshold_tss_hss)>1:
        best_xss_threshold_tss_hss = best_xss_threshold_tss_hss[0]
        Y_best_predicted_tss_hss = probability_prediction > best_xss_threshold_tss_hss
    else:
        Y_best_predicted_tss_hss = probability_prediction > best_xss_threshold_tss_hss
    print('best (TSS+HSS)/2')
    metrics_training_tss_hss = metrics_classification_weight(Y_training > 0, Y_best_predicted_tss_hss)
    


    return best_xss_threshold, metrics_training, nss_vector, best_xss_threshold_tss, best_xss_threshold_hss, best_xss_threshold_tss_hss, np.max(comb_tss_hss)



def metrics_classification_weight(y_real, y_pred, print_skills=True):

    cm, far, pod, acc, hss, tss, fnfp, csi = classification_skills_weight(y_real, y_pred)

    if print_skills:
        print ('confusion matrix')
        print (cm)
        print ('false alarm ratio       \t', far)
        print ('probability of detection\t', pod)
        print ('accuracy                \t', acc)
        print ('hss                     \t', hss)
        print ('tss                     \t', tss)
        print ('balance                 \t', fnfp)
        print ('csi                 \t', csi)

    balance_label = float(sum(y_real)) / y_real.shape[0]

    #cm, far, pod, acc, hss, tss, fnfp = classification_skills(y_real, y_pred)

    return {
        "cm": cm,
        "far": far,
        "pod": pod,
        "acc": acc,
        "hss": hss,
        "tss": tss,
        "fnfp": fnfp,
        "balance label": balance_label,
        "csi": csi}

def classification_skills_weight(y_real, y_pred):

    TN,FP,FN,TP = weighted_confusion_matrix(y_real, y_pred)

    if (TP + FP + FN + TN) == 0.:
        if (TP + TN) == 0.:
            acc = 0.  # float('NaN')
        else:
            acc = -100  # float('Inf')
    else:
        acc = (TP + TN) / (TP + FP + FN + TN)

    if TP + FN == 0.:
        if TP == 0.:
            tss_aux1 = 0.  # float('NaN')
        else:
            tss_aux1 = -100  # float('Inf')
    else:
        tss_aux1 = (TP / (TP + FN))

    if (FP + TN) == 0.:
        if FP == 0.:
            tss_aux2 = 0.  # float('NaN')
        else:
            tss_aux2 = -100  # float('Inf')
    else:
        tss_aux2 = (FP / (FP + TN))

    tss = tss_aux1 - tss_aux2

    if ((TP + FN) * (FN + TN) + (TP + FP) * (FP + TN)) == 0.:
        if (TP * TN - FN * FP) == 0:
            hss = 0.  # float('NaN')
        else:
            hss = -100  # float('Inf')
    else:
        hss = 2 * (TP * TN - FN * FP) / ((TP + FN) *
                                         (FN + TN) + (TP + FP) * (FP + TN))

    if FP == 0.:
        if FN == 0.:
            fnfp = 0.  # float('NaN')
        else:
            fnfp = -100  # float('Inf')
    else:
        fnfp = FN / FP

    if (TP + FN) == 0.:
        if TP == 0.:
            pod = 0  # float('NaN')
        else:
            pod = -100  # float('Inf')
    else:
        pod = TP / (TP + FN)


    if (TP + FP) == 0.:
        if FP == 0.:
            far = 0.  # float('NaN')
        else:
            far = -100  # float('Inf')
    else:
        far = FP / (TP + FP)

    #acc = (a + d) / (a + b + c + d)
    #tpr = a / (a + b)
    #tnr = d / (d + c)
    #wtpr = a / (a + b) * (a + c) / (a + b + c + d) + d / (c + d) * (b + d) / (a + b + c + d)
    #pacc = a / (a + c)
    #nacc = d / (b + d)
    #wacc = a / (a + c) * (a + c) / (a + b + c + d) + d / (b + d) * (b + d) / (a + b + c + d)

    # if the cm has a row or a column equal to 0, we have bad tss
    if TP+FN == 0 or TN+FP == 0 or TP+FP == 0 or TN+FN == 0:
        tss = 0
        
    if TP+FP+FN==0:
        csi = 0
    else:
        csi = TP/(TP+FP+FN)
    
    
    weighted_cm = np.zeros((2,2))
    weighted_cm[0,0]=TN
    weighted_cm[0,1]=FP
    weighted_cm[1,0]=FN
    weighted_cm[1,1]=TP
    #, pod, acc, hss, tss, fnfp, tpr, tnr, pacc, nacc, wacc, wtpr)
    return weighted_cm, far, pod, acc, hss, tss, fnfp, csi

def weighted_confusion_matrix(y_true, y_pred):
    TP_values = np.logical_and(np.equal(y_true, True), np.equal(y_pred, True))
    idx_TP=np.where(TP_values==True)
    TP=len(idx_TP[0])
    
    TN_values = np.logical_and(np.equal(y_true, False), np.equal(y_pred, False))
    idx_TN=np.where(TN_values==True)
    TN=len(idx_TN[0])


    FP_values = np.logical_and(np.equal(y_true, False), np.equal(y_pred, True))
    idx_FP=np.where(FP_values==True)
    mask = [1./2.,1./3.,1./4.]
    FP=0
    window_hour=3
    if y_true.shape[0] >=6:
    
        for t in idx_FP[0]: 
            if t >=  window_hour and t <= len(y_true)- window_hour-1:
               
                y_true_window = y_true[t- window_hour:t+ window_hour+1]
                y_pred_window = y_pred[t- window_hour:t+ window_hour+1]
        
                if len(np.where(y_true_window==1)[0]) >= 1:
                    count_FP = 1-np.max(mask*y_true[t+1:t+ window_hour+1])
                else:
                    count_FP = 2
            elif t<window_hour:
                y_true_window = y_true[:t+ window_hour+1]
                y_pred_window = y_pred[:t+ window_hour+1]
                if len(np.where(y_true_window==1)[0]) >= 1:
                    count_FP = 1-np.max(mask*y_true[t+1:t+ window_hour+1])
                else:
                    count_FP = 2
            elif t > len(y_true)-window_hour-1:
                y_true_window = y_true[t- window_hour:]
                y_pred_window = y_pred[t- window_hour:]
                if len(np.where(y_true_window==1)[0]) >= 1:
                    if t < len(y_true)-1:
                        count_FP = 1-np.max(mask[0:len(y_true)-t-1]*y_true[t+1:])
                    elif t == len(y_true)-1:
                        count_FP = 1
                
                else:
                    count_FP = 2
            FP=FP+count_FP
    
    if y_true.shape[0]<6:
        for t in idx_FP[0]: 
            if t<window_hour:
                if t+window_hour+1<=np.shape(y_true)[0]:
                    y_true_window = y_true[:t+ window_hour+1]
                    y_pred_window = y_pred[:t+ window_hour+1]
                    if len(np.where(y_true_window==1)[0]) >= 1:
                        count_FP = 1-np.max(mask*y_true[t+1:t+ window_hour+1])
                    else:
                        count_FP = 2
                elif t+window_hour+1 >np.shape(y_true)[0]:
                    y_true_window = y_true
                    y_pred_window = y_pred
                    if len(np.where(y_true_window==1)[0]) >= 1:
                        if t < len(y_true)-1:
                            count_FP = 1-np.max(mask[0:len(y_true)-t-1]*y_true[t+1:])
                        elif t == len(y_true)-1:
                            count_FP = 1
                
                    else:
                        count_FP = 2
            elif t > len(y_true)-window_hour-1:
                if t- window_hour>=0:
                    y_true_window = y_true[t- window_hour:]
                    y_pred_window = y_pred[t- window_hour:]
                    if len(np.where(y_true_window==1)[0]) >= 1: 
                        if t < len(y_true)-1:
                            count_FP = 1-np.max(mask[0:len(y_true)-t-1]*y_true[t+1:])
                        elif t == len(y_true)-1:
                            count_FP = 1
                
                    else:
                        count_FP = 2
                elif t- window_hour<0:
                    y_true_window = y_true
                    y_pred_window = y_pred
                    if len(np.where(y_true_window==1)[0]) >= 1:
                        if t < len(y_true)-1:
                            count_FP = 1-np.max(mask[0:len(y_true)-t-1]*y_true[t+1:])
                        elif t == len(y_true)-1:
                            count_FP = 1
                
                    else:
                        count_FP = 2
            FP=FP+count_FP
                        
        
    FN_values = np.logical_and(np.equal(y_true, True), np.equal(y_pred, False))
    idx_FN=np.where(FN_values==True)
    FN=0
    mask_FN=[1./4.,1./3.,1./2.]
    if y_true.shape[0]>=6:
        for t in idx_FN[0]: 

            if t >=  window_hour and t <= len(y_true)- window_hour-1:
                
                y_true_window = y_true[t- window_hour:t+ window_hour+1]
                y_pred_window = y_pred[t- window_hour:t+ window_hour+1]
                if len(np.where(y_pred_window==1)[0]) >= 1:
                    count_FN = 1-np.max(mask_FN*y_pred[t- window_hour:t])
                else:
                    count_FN = 2
            elif t<window_hour:
                y_true_window = y_true[:t+ window_hour+1]
                y_pred_window = y_pred[:t+ window_hour+1]
                if len(np.where(y_pred_window==1)[0]) >= 1:
                    if t > 0:
                        count_FN = 1-np.max(mask_FN[window_hour-t:window_hour]*y_pred[:t])
                    elif t ==0:
                        count_FN = 1
                else:
                    count_FN = 2
            elif t > len(y_true)- window_hour-1:
                y_true_window = y_true[t- window_hour:]
                y_pred_window = y_pred[t- window_hour:]
                if len(np.where(y_pred_window==1)[0]) >= 1:
                    count_FN = 1-np.max(mask_FN*y_pred[t- window_hour:t])
                else:
                    count_FN = 2

            FN=FN+count_FN
                
                
    if y_true.shape[0]<6:
        for t in idx_FN[0]:
            if t<window_hour:
                if t+window_hour+1<=np.shape(y_true)[0]:
                    y_true_window = y_true[:t+ window_hour+1]
                    y_pred_window = y_pred[:t+ window_hour+1]
                    if len(np.where(y_pred_window==1)[0]) >= 1:
                        if t > 0:
                            count_FN = 1-np.max(mask_FN[window_hour-t:window_hour]*y_pred[:t])
                        elif t ==0:
                            count_FN = 1
                    else:
                        count_FN = 2
                elif t+window_hour+1 >np.shape(y_true)[0]:
                    y_true_window = y_true
                    y_pred_window = y_pred
                    if len(np.where(y_pred_window==1)[0]) >= 1:
                        if t > 0:
                            count_FN = 1-np.max(mask_FN[window_hour-t:window_hour]*y_pred[:t])
                        elif t == 0:
                            count_FN = 1
                    else:
                        count_FN = 2
                            
            elif t > len(y_true)-window_hour-1:
                if t- window_hour>=0:
                    y_true_window = y_true[t- window_hour:]
                    y_pred_window = y_pred[t- window_hour:]
                    if len(np.where(y_pred_window==1)[0]) >= 1:
                        count_FN = 1-np.max(mask_FN*y_pred[t- window_hour:t])
                    else:
                        count_FN = 2
                elif t- window_hour<0:
                    y_true_window = y_true
                    y_pred_window = y_pred
                    if len(np.where(y_pred_window==1)[0]) >= 1:
                        if t > 0:
                            count_FN = 1-np.max(mask_FN[window_hour-t:window_hour]*y_pred[:t])
                        elif t == 0:
                            count_FN = 1
                    else:
                        count_FN = 2

            FN=FN+count_FN
            
            
    return TN, FP, FN, TP


def weighted_confusion_matrix_old(y_true, y_pred):
    TP_values = np.logical_and(np.equal(y_true, True), np.equal(y_pred, True))
    idx_TP=np.where(TP_values==True)
    TP=len(idx_TP[0])
    
    TN_values = np.logical_and(np.equal(y_true, False), np.equal(y_pred, False))
    idx_TN=np.where(TN_values==True)
    TN=len(idx_TN[0])


    FP_values = np.logical_and(np.equal(y_true, False), np.equal(y_pred, True))
    idx_FP=np.where(FP_values==True)
    mask = [1./2.,1./3.,1./4.]
    FP=0
    window_hour=3
    
    for t in idx_FP[0]: #range(len(y_true)):
        if t >=  window_hour and t <= len(y_true)- window_hour-1:
            #window -4 +4
            y_true_window = y_true[t- window_hour:t+ window_hour+1]
            y_pred_window = y_pred[t- window_hour:t+ window_hour+1]
        
            if len(np.where(y_true_window==1)[0]) >= 1:
                count_FP = 1-np.max(mask*y_true[t+1:t+ window_hour+1])
            else:
                count_FP = 2
        elif t<window_hour:
            y_true_window = y_true[:t+ window_hour+1]
            y_pred_window = y_pred[:t+ window_hour+1]
            if len(np.where(y_true_window==1)) > 1:
                count_FP = 1-np.max(mask*y_true[t+1:t+ window_hour+1])
            else:
                count_FP = 2
        elif t > len(y_true)-window_hour-1:
            y_true_window = y_true[t- window_hour:]
            y_pred_window = y_pred[t- window_hour:]
            if len(np.where(y_true_window==1)) > 1:
                if t < len(y_true)-1:
                    count_FP = 1-np.max(mask[0:len(y_true)-t-1]*y_true[t+1:])
                elif t == len(y_true)-1:
                    count_FP = 1
                
            else:
                count_FP = 2
        FP=FP+count_FP
    

    FN_values = np.logical_and(np.equal(y_true, True), np.equal(y_pred, False))
    idx_FN=np.where(FN_values==True)
    FN=0
    mask_FN=[1./4.,1./3.,1./2.]
    for t in idx_FN[0]: 

        if t >=  window_hour and t <= len(y_true)- window_hour-1:

            y_true_window = y_true[t- window_hour:t+ window_hour+1]
            y_pred_window = y_pred[t- window_hour:t+ window_hour+1]
            if len(np.where(y_pred_window==1)[0]) >= 1:
                count_FN = 1-np.max(mask_FN*y_pred[t- window_hour:t])
            else:
                count_FN = 2
        elif t<window_hour:
            y_true_window = y_true[:t+ window_hour+1]
            y_pred_window = y_pred[:t+ window_hour+1]
            if len(np.where(y_pred_window==1)) > 1:
                if t > 0:
                    count_FN = 1-np.max(mask_FN[t-1:2]*y_pred[:t])
                elif t ==0:
                    count_FN = 1
            else:
                count_FN = 2
        elif t > len(y_true)- window_hour-1:
            y_true_window = y_true[t- window_hour:]
            y_pred_window = y_pred[t- window_hour:]
            if len(np.where(y_pred_window==1)) > 1:
                count_FN = 1-np.max(mask_FN*y_pred[t- window_hour:t])
            else:
                count_FN = 2
        FN=FN+count_FN

    return TN, FP, FN, TP


def weighted_confusion_matrix_threshold(y_true, y_pred, threshold):
    y_pred = y_pred>threshold
    TP_values = np.logical_and(np.equal(y_true, True), np.equal(y_pred, True))
    idx_TP=np.where(TP_values==True)
    TP=len(idx_TP[0])
    
    TN_values = np.logical_and(np.equal(y_true, False), np.equal(y_pred, False))
    idx_TN=np.where(TN_values==True)
    TN=len(idx_TN[0])


    FP_values = np.logical_and(np.equal(y_true, False), np.equal(y_pred, True))
    idx_FP=np.where(FP_values==True)
    mask = [1./2.,1./3.,1./4.]
    FP=0
    window_hour=3

    for t in idx_FP[0]: 
        if t >=  window_hour and t <= len(y_true)- window_hour-1:

            y_true_window = y_true[t- window_hour:t+ window_hour+1]
            y_pred_window = y_pred[t- window_hour:t+ window_hour+1]
        
            if len(np.where(y_true_window==1)[0]) >= 1:
                count_FP = 1-np.max(mask*y_true[t+1:t+ window_hour+1])
            else:
                count_FP = 2
        elif t<window_hour:
            y_true_window = y_true[:t+ window_hour+1]
            y_pred_window = y_pred[:t+ window_hour+1]
            if len(np.where(y_true_window==1)) > 1:
                count_FP = 1-np.max(mask*y_true[t+1:t+ window_hour+1])
            else:
                count_FP = 2
        elif t > len(y_true)-window_hour-1:
            y_true_window = y_true[t- window_hour:]
            y_pred_window = y_pred[t- window_hour:]
            if len(np.where(y_true_window==1)) > 1:
                if t < len(y_true)-1:
                    count_FP = 1-np.max(mask[0:len(y_true)-t-1]*y_true[t+1:])
                elif t == len(y_true)-1:
                    count_FP = 1
                
            else:
                count_FP = 2
        FP=FP+count_FP
    

    FN_values = np.logical_and(np.equal(y_true, True), np.equal(y_pred, False))
    idx_FN=np.where(FN_values==True)
    FN=0
    mask_FN=[1./4.,1./3.,1./2.]
    for t in idx_FN[0]: 

        if t >=  window_hour and t <= len(y_true)- window_hour-1:

            y_true_window = y_true[t- window_hour:t+ window_hour+1]
            y_pred_window = y_pred[t- window_hour:t+ window_hour+1]
            if len(np.where(y_pred_window==1)[0]) >= 1:
                count_FN = 1-np.max(mask_FN*y_pred[t- window_hour:t])
            else:
                count_FN = 2
        elif t<window_hour:
            y_true_window = y_true[:t+ window_hour+1]
            y_pred_window = y_pred[:t+ window_hour+1]
            if len(np.where(y_pred_window==1)) > 1:
                if t > 0:
                    count_FN = 1-np.max(mask_FN[t-1:2]*y_pred[:t])
                elif t ==0:
                    count_FN = 1
            else:
                count_FN = 2
        elif t > len(y_true)- window_hour-1:
            y_true_window = y_true[t- window_hour:]
            y_pred_window = y_pred[t- window_hour:]
            if len(np.where(y_pred_window==1)) > 1:
                count_FN = 1-np.max(mask_FN*y_pred[t- window_hour:t])
            else:
                count_FN = 2
        FN=FN+count_FN

    return TN, FP, FN, TP


#***
def training_set_standardization(X_training):
    # standardization
    mean_ = X_training.sum(axis=0) / X_training.shape[0]


    std_ = np.sqrt(
        (((X_training - mean_) ** 2.).sum(axis=0) / X_training.shape[0]))

    Xn_training = div0((X_training - mean_), std_)

    return Xn_training, mean_, std_

def testing_set_standardization(X_testing, mean_, std_):
    Xn_testing = div0((X_testing - mean_), std_)

    return Xn_testing

def div0(a, b):
    """
    ignore / 0, div0( [-1, 0, 1], 0 ) -> [0, 0, 0]

    Parameters
    ----------
    a
    b

    Returns
    -------

    c

    """
    with np.errstate(divide='ignore', invalid='ignore'):
        c = np.true_divide(a, b)
        c[~ np.isfinite(c.tolist())] = 0  # -inf inf NaN
    return c





def compute_weight_cm_tss_threshold(y, pred,threshold):
    # Compute confusion matrix and TSS given y_true and y_pred in categorical form
    TN,FP,FN,TP = weighted_confusion_matrix_threshold(y,pred,threshold)

    if TP + FN == 0.:
        if TP == 0.:
            tss_aux1 = 0.  # float('NaN')
        else:
            tss_aux1 = -100  # float('Inf')
    else:
        tss_aux1 = (TP / (TP + FN))

    if (FP + TN) == 0.:
        if FP == 0.:
            tss_aux2 = 0.  # float('NaN')
        else:
            tss_aux2 = -100  # float('Inf')
    else:
        tss_aux2 = (FP / (FP + TN))

    tss = tss_aux1 - tss_aux2
    
    if ((TP + FN) * (FN + TN) + (TP + FP) * (FP + TN)) == 0.:
        if (TP * TN - FN * FP) == 0:
            hss = 0.  # float('NaN')
        else:
            hss = -100  # float('Inf')
    else:
        hss = 2 * (TP * TN - FN * FP) / ((TP + FN) *
                                         (FN + TN) + (TP + FP) * (FP + TN))
    
    if TP+FP+FN==0:
        CSI=0
    else:
        CSI = TP/(TP+FP+FN)
    
    weighted_cm = np.zeros((2,2))
    weighted_cm[0,0]=TN
    weighted_cm[0,1]=FP
    weighted_cm[1,0]=FN
    weighted_cm[1,1]=TP
    
    return weighted_cm, tss, hss, CSI

def predict_ensemble(tss_val_array,perc,list_epochs,X_test_std,opt_threshold):
    
    pred_0_1_test_list=[]
    alpha=np.max(tss_val_array)*perc
    
    
    idx=np.where(np.array(tss_val_array)>alpha)
    idx=idx[0]
    
    for i in idx:
        file=list_epochs[i]
        #print(file)
        model = load_model(file,compile=False)
        pred_test = model.predict(X_test_std)
        pred_prob_test = pred_test.reshape(1,len(pred_test))
        pred_prob_test = pred_prob_test[0]
        
        pred_0_1_test = pred_prob_test > opt_threshold[i]
        pred_0_1_test_list.append(pred_0_1_test)
    
    
    pred_0_1_arr=np.array(pred_0_1_test_list)*1
   
    
    pred_median_pred_0_1=np.median(pred_0_1_arr,axis=0)
   
    idx_to_discard=np.where(pred_median_pred_0_1==0.5)[0]
    pred_median_pred_0_1[idx_to_discard]=1
    
    return pred_median_pred_0_1


def select_best_patience_on_val(folder_path,train_X,train_y, val_X, val_y):
    #Load models obtained by applying early stopping with 5 different patiences (10,20,30,40,50)
    #For each model compute the best thresholds by optimizing TSS and wTSS
    # Choose models with the highest TSS and the highest wTSS in validation
    num_patience=5
    tss_opt_tss = np.zeros(num_patience)
    threshold_opt_tss = np.zeros(num_patience)
    wtss_opt_wtss = np.zeros(num_patience)
    threshold_opt_wtss = np.zeros(num_patience)
    for i in range(num_patience):
        patience=(i+1)*10
        model = Sequential()
        model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
        model.add(Dense(1,activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam')
        model.load_weights(folder_path+'early_stop'+str(patience)+'_lstm_1times_pred_poll_8features_train_13104_val_4320_poll_450_epoch.hdf5')
        #predict on training in order to compute the optimum threshold
        y_pred_train=model.predict(train_X)
        pred_prob = y_pred_train.reshape(1,len(y_pred_train))
        pred_prob = pred_prob[0]
    
        threshold_nss, metrics_training, nss_vector, threshold_tss, threshold_hss, threshold_tss_hss, max_tss_hss = optimize_threshold_skill_scores(pred_prob, train_y)
        print(threshold_tss)
        threshold_opt_tss[i]=threshold_tss
        threshold_nss_w, metrics_training_w, nss_vector_w, threshold_tss_weight, threshold_hss_weight, threshold_tss_hss_weight, max_tss_hss_weight = optimize_threshold_skill_scores_weight_matrix(pred_prob, train_y)
        print(threshold_tss_weight)
        threshold_opt_wtss[i]=threshold_tss_weight
    
        #predict on validation in order to take the choose the best patience
        pred_val=model.predict(val_X)
        pred_val=pred_val.reshape(1,len(pred_val))
        pred_val=pred_val[0]
    
        #print('results on validation with tss optimization') 
        cm_val, tss_val, hss_val, csi_val = compute_cm_tss_threshold(val_y, pred_val,threshold_tss)#_weight
        #print(cm_val)
        #print('tss = ','{:0.4f}'.format(tss_val))
        #print('hss = ','{:0.4f}'.format(hss_val))
        #print('csi = ','{:0.4f}'.format(csi_val))
        tss_opt_tss[i]=tss_val
        #wcm_val, wtss_val, whss_val, wcsi_val = compute_weight_cm_tss(val_y, pred_val>threshold_tss)#_weight
        #print('Weighted Skill scores')
        #print(wcm_val)
        #print('wtss = ','{:0.4f}'.format(wtss_val))
        #print('whss = ','{:0.4f}'.format(whss_val))
        #print('wcsi = ','{:0.4f}'.format(wcsi_val))
    
        #print('results on validation with wtss optimization') 
        #cm_val, tss_val, hss_val, csi_val = compute_cm_tss_threshold(val_y, pred_val,threshold_tss_weight)#_weight
        #print(cm_val)
        #print('tss = ','{:0.4f}'.format(tss_val))
        #print('hss = ','{:0.4f}'.format(hss_val))
        #print('csi = ','{:0.4f}'.format(csi_val))
        wcm_val, wtss_val, whss_val, wcsi_val = compute_weight_cm_tss(val_y, pred_val>threshold_tss_weight)#_weight
        #print('Weighted Skill scores')
        #print(wcm_val)
        #print('wtss = ','{:0.4f}'.format(wtss_val))
        #print('whss = ','{:0.4f}'.format(whss_val))
        #print('wcsi = ','{:0.4f}'.format(wcsi_val))
        wtss_opt_wtss[i]=wtss_val
        
    return tss_opt_tss,wtss_opt_wtss, threshold_opt_tss, threshold_opt_wtss



def predict(folder_path,train_X_shape_1,train_X_shape_2,idx_patience_tss,test_X,threshold_opt_tss):
    
    model = Sequential()
    model.add(LSTM(50, input_shape=(train_X_shape_1, train_X_shape_2)))
    model.add(Dense(1,activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam')#temporal_binary_cross
    model.load_weights(folder_path+'early_stop'+str(int((idx_patience_tss+1)*10))+'_lstm_1times_pred_poll_8features_train_13104_val_4320_poll_450_epoch.hdf5')
    #predict on test
    pred_prob_test=model.predict(test_X)
    y_pred_test = pred_prob_test>threshold_opt_tss[idx_patience_tss]
    y_pred_test = y_pred_test*1
    y_pred_test = y_pred_test[:,0]
    
    return y_pred_test


        
    
def make_box_figure(y_pred_test_box1,tt1):
    
    idx_pred_yes_isolated = [j for j in range(1,y_pred_test_box1.shape[0]-1) if (y_pred_test_box1[j-1]==0 and y_pred_test_box1[j]==1 and y_pred_test_box1[j+1]==0)]
    idx_pred_no_isolated = [j for j in range(1,y_pred_test_box1.shape[0]-1) if (y_pred_test_box1[j-1]==1 and y_pred_test_box1[j]==0 and y_pred_test_box1[j+1]==1)]

    y_pred_test_box1_new=[]

    tt1_new=[]
    for idx in range(len(y_pred_test_box1)):
        y_pred_test_box1_new.append(y_pred_test_box1[idx])
        tt1_new.append(tt1[idx])
        if idx in idx_pred_yes_isolated:
            y_pred_test_box1_new.append(1)
            tt1_new.append(tt1[idx]+dt.timedelta(seconds=1))
        if idx in idx_pred_no_isolated:
            y_pred_test_box1_new.append(0)
            tt1_new.append(tt1[idx]+dt.timedelta(seconds=1))
    y_pred_test_box1_new=np.array(y_pred_test_box1_new)

    tt1_new=sorted(tt1_new) 

    tt1_new_arrange = tt1_new
    for j in range(0,y_pred_test_box1_new.shape[0]-1):
        if (y_pred_test_box1_new[j]==0 and y_pred_test_box1_new[j+1]==1):
            tt1_new_arrange[j]=tt1_new[j]+dt.timedelta(minutes=39)
            tt1_new_arrange[j+1]=tt1_new[j+1]-dt.timedelta(minutes=20)
        if (y_pred_test_box1_new[j]==1 and y_pred_test_box1_new[j+1]==0):
            tt1_new_arrange[j]=tt1_new[j]+dt.timedelta(minutes=19)
            tt1_new_arrange[j+1]=tt1_new[j+1]-dt.timedelta(minutes=40)
            
    return tt1_new_arrange, y_pred_test_box1_new

        
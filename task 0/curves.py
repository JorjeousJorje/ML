import numpy as np

from metrics import TPR, FPR, PPV
from scipy.integrate import trapz 

def roc_curve(y_test, y_score):
    fprs, tprs, thresholds = [], [], np.unique(y_score)[::-1]
    for th in thresholds:
        tprs.append(TPR(probs=y_score, ground=y_test, thresh=th, roc=True))
        fprs.append(FPR(probs=y_score, ground=y_test, thresh=th, roc=True))
    
    return np.asarray(fprs), np.asarray(tprs), np.asarray(thresholds)

def auc(fpr, tpr):
    AUC = trapz(tpr, fpr)
    return AUC 
        
def precision_recall_curve(y_test, y_score):
    precision, recall, thresholds = [], [], np.unique(y_score)[::-1]
    thresholds.sort()
    for th in thresholds:
        
        tpr = TPR(probs=y_score, ground=y_test, thresh=th)
        ppv = PPV(probs=y_score, ground=y_test, thresh=th)
    
        recall.append(tpr)
        precision.append(ppv)
        
    recall.append(0)
    precision.append(1)
        
    return np.asarray(precision), np.asarray(recall), np.asarray(thresholds)




















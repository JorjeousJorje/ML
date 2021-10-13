import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapz 


def TP(probs, ground, thresh):
    return np.sum((probs > thresh) & (ground == 1))

def FP(probs, ground, thresh):
    return np.sum((probs > thresh) & (ground == 0))

def TN(probs, ground, thresh):
    return np.sum((probs < thresh) & (ground == 0))

def FN(probs, ground, thresh):
    return np.sum((probs < thresh) & (ground == 1))


# Recall
def TPR(probs, ground, thresh):
    tp = TP(probs, ground, thresh)
    fn = FN(probs, ground, thresh)
    return tp / (tp + fn)

def TNR(probs, ground, thresh):
    tn = TN(probs, ground, thresh)
    fp = FP(probs, ground, thresh)
    return tn / (tn + fp)

def FPR(probs, ground, thresh):
    return 1.0 - TNR(probs, ground, thresh)

# Presicion
def PPV(probs, ground, thresh):
    fp = FP(probs, ground, thresh)
    tp = TP(probs, ground, thresh)
    
    if np.isclose(fp, 0.0):
        return 1.0
    return tp / (tp + fp)


def roc_curve(y_test, y_score):
    fprs, tprs, thresholds = [], [], np.unique(y_score)[::-1]
    for th in thresholds:
        tprs.append(TPR(probs=y_score, ground=y_test, thresh=th))
        fprs.append(FPR(probs=y_score, ground=y_test, thresh=th))
    
    return fprs, tprs, thresholds


def auc(fpr, tpr):
    AUC = trapz(tpr, fpr)
    return AUC 


def plot_roc(fpr, tpr, title: str):
    with plt.style.context(['science']):
        plt.figure(dpi=100)
        plt.title(title) 
        plt.xlabel('FPR', fontsize=10)
        plt.ylabel('TPR', fontsize=10)
        plt.plot(fpr, tpr, label='ROC Curve')
        plt.legend()
        plt.show()
        
def plot_pr(tpr, ppv,  title: str):
    with plt.style.context(['science']): 
        plt.figure(dpi=100)
        plt.title(title) 
        plt.xlabel('TPR', fontsize=10)
        plt.ylabel('PPV', fontsize=10)
        plt.plot(tpr, ppv, label='PR Curve')
        plt.legend()
        plt.show()


def precision_recall_curve(y_test, y_score):
    # test this
    precision, recall, thresholds = [], [], np.unique(y_score)[::-1]
    for th in thresholds:
        recall.append(TPR(probs=y_score, ground=y_test, thresh=th))
        precision.append(PPV(probs=y_score, ground=y_test, thresh=th))
    
    return precision, recall, thresholds




















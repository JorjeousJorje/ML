import numpy as np


def TPRoc(probs, ground, thresh):
    return np.sum((probs > thresh) & (ground == 1))

def FPRoc(probs, ground, thresh):
    return np.sum((probs > thresh) & (ground == 0))

def TNRoc(probs, ground, thresh):
    return np.sum((probs < thresh) & (ground == 0))

def FNRoc(probs, ground, thresh):
    return np.sum((probs < thresh) & (ground == 1))

def TP(probs, ground, thresh):
    return np.sum((probs >= thresh) & (ground == 1))

def FP(probs, ground, thresh):
    return np.sum((probs >= thresh) & (ground == 0))

def TN(probs, ground, thresh):
    return np.sum((probs <= thresh) & (ground == 0))

def FN(probs, ground, thresh):
    return np.sum((probs < thresh) & (ground == 1))

# Recall
def TPR(probs, ground, thresh, roc=False):
    tp = TP(probs, ground, thresh)
    fn = FN(probs, ground, thresh)
    if roc:
        tp = TPRoc(probs, ground, thresh)
        fn = FNRoc(probs, ground, thresh)
    return tp / (tp + fn)

def TNR(probs, ground, thresh, roc=False):
    tn = TN(probs, ground, thresh)
    fp = FP(probs, ground, thresh)
    if roc:
        tn = TNRoc(probs, ground, thresh)
        fp = FPRoc(probs, ground, thresh)
    return tn / (tn + fp)

def FPR(probs, ground, thresh, roc=False):
    return 1.0 - TNR(probs, ground, thresh, roc)

# Presicion
def PPV(probs, ground, thresh):
    fp = FP(probs, ground, thresh)
    tp = TP(probs, ground, thresh)
    return tp / (tp + fp)
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score
METRICS=['acc','balanced_acc','f1_macro','recall_N','recall_IR','recall_OR','recall_B']
KEYS=['split','source_load_hp','target_load_hp','protocol','method','seed','condition']
LABELS=['N','IR','OR','B']


def scores(labels,p):
    counts=confusion_matrix(labels,p.argmax(1),labels=range(4))
    support=counts.sum(1)
    if (support==0).any():
        raise ValueError('测试缺失类别')
    recall=counts.diagonal()/support
    return dict(acc=float(np.trace(counts)/counts.sum()),balanced_acc=float(recall.mean()),
        f1_macro=float(f1_score(labels,p.argmax(1),labels=range(4),average='macro',zero_division=0)),
        **dict(zip(METRICS[3:],map(float,recall)))),counts


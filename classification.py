import numpy as np

def calc_AP(predictions, labels):
    assert len(predictions)==len(labels)

    sortind = np.argsort(-predictions)
    tp = labels[sortind]==1
    fp = labels[sortind]!=1
    npos = np.sum(labels)

    fp = np.cumsum(fp).astype('float32')
    tp = np.cumsum(tp).astype('float32')
    rec = tp/npos
    prec = tp/(fp+tp)
    tmp = (labels[sortind]==1).astype('float32')

    return np.sum(tmp*prec)/npos

def calc_classification_mAP(predictions, labels):
    AP = []
    num_class = labels.shape[1]
    for i in range(num_class):
       if np.sum(labels[:,i]) > 0:
           AP.append(calc_AP(predictions[:,i], labels[:,i]))

    return 100*sum(AP)/len(AP)

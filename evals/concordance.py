import numpy as np
from lifelines.metrics import concordance_index as C_index

def concordance_index(y_true, y_pred):
    '''
    Calculate the concordance index for regular DeepSurv tasks.
    :param y_true: numpy.array
           Observed survival time.Negative values represent right censored.
    :param y_pred: numpy.array
           Predicted survival time.
    :return: float
           Concordance index
    '''
    y_true = np.squeeze(y_true)
    y_pred = np.squeeze(y_pred)
    T = np.abs(y_true)
    E = (y_true>0).astype(np.int64)
    return C_index(T, y_pred, E)

def c_index_deephit(Prediction, Time_survival, Death, Time):
    '''
    This is a cause-specific c(t)-index
    :param Prediction: numpy.array
           Risk at Time (higher value means more risky).
    :param Time_survival: numpy.array
           Survival/censoring time.
    :param Death: numpy.array
           `Death==1` means death, `Death==0` means censoring or death from other causes.
    :param Time: numpy.array
           Time of evaluation (time-horizon when evaluating C-index)
    :return: float
           The cause-specific concordance index for DeepHit tasks.
    '''

    N = len(Prediction)
    A = np.zeros((N,N))
    Q = np.zeros((N,N))
    N_t = np.zeros((N,N))
    Num = 0
    Den = 0
    for i in range(N):
        A[i, np.where(Time_survival[i] < Time_survival)] = 1
        Q[i, np.where(Prediction[i] > Prediction)] = 1
  
        if (Time_survival[i]<=Time and Death[i]==1):
            N_t[i,:] = 1

    Num  = np.sum(((A)*N_t)*Q)
    Den  = np.sum((A)*N_t)

    if Num == 0 and Den == 0:
        result = -1
    else:
        result = float(Num/Den)

    return result
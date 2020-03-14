'''
The Brier score is frequently used by meteorologists to measure the skill of binary probabilistic forecasts.
For more information about Brier score, please refer to

Jewson, Stephen. (2004). The problem with the Brier score.
'''

import numpy as np

def brier_score(pred, time_survival, death, time):
    N = pred.shape[0]
    y_true = ((time_survival <= time) * death).astype(float)
    return np.mean((pred - y_true)**2)
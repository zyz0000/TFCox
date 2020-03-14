import numpy as np
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.plotting import add_at_risk_counts
from lifelines.utils import concordance_index as C_index

def check_config(config):
    '''
    To check the input parameters for training a neural network.
    :param config: Dictionary.
           The input parameters for training neural network.
    :return: Corrected configuration.
    '''
    default_network_config = {
        'learning_rate' : 0.01,
        'learning_rate_decay': 1.0,
        'activation': 'relu',
        'L2_reg': 0.0,
        'L1_reg': 0.0,
        'optimizer': 'adam',
        'dropout_rate': 1.0,
        'seed': 0
    }
    for item in default_network_config.keys():
        if item not in config:
            config[item] = default_network_config[item]


def check_surv_data(X, y):
    '''
    To check that if the inputs are the form of right censored data.
    :param X: pandas.DataFrame
           Covariates.
    :param y: pandas.DataFrame
           Labels of survival data. Negative values are considered as right censored.
    '''
    if not isinstance(X, pd.DataFrame):
        raise TypeError('The type of input covariates X must be pandas.DataFrame')
    if not isinstance(y, pd.DataFrame):
        raise TypeError('The type of input covariates y must be pandas.DataFrame')
    if y.shape[1] != 1:
        raise TypeError('The number of columns of y must be 1 but got {0}'.format(y.shape[1]))


def prepare_surv_data(X, y):
    '''
    :param X: pandas.DataFrame
           Covariates.
    :param y: pandas.DataFrame
           Labels of survival data. Negative values are considered as right censored.
    :return: pandas.DataFrame
           A standard survival analysis dataframe.
    '''
    check_surv_data(X, y)
    T = -np.abs(np.squeeze(np.array(y)))
    sorted_idx = np.argsort(T)
    return sorted_idx, X.iloc[sorted_idx,:], y.iloc[sorted_idx,:]


def baseline_hazard(label_E, label_T, pred_hr):
    ind_df = pd.DataFrame(
        {'E': label_E,
         'T': label_T,
         'P': pred_hr}
    )
    sums_over_durations = ind_df.groupby('T')[['P', 'E']].sum()
    # Sort the predicted hazard rate in descending order and cumsum it
    sums_over_durations['P'] = sums_over_durations['P'].loc[::-1].cumsum()
    baseline_hazard = pd.DataFrame(
        {'baseline_hazard': (sums_over_durations['E'] / sums_over_durations['P'])}
    )
    return baseline_hazard

def baseline_cumulative_hazard(label_E, label_T, pred_hr):
    '''
    Calculate the cumulative baseline hazard, that is $\Lambda_0(t)=\int_{0}^{t} \lambda_0(u) du$
    :param label_E: numpy.array
           Indicators to indicate whether right censored or not.
    :param label_T: numpy.array
           Observed survival time.Negative values represent right censored.
    :param pred_hr: numpy.array
           Predicted hazard ratio.
    :return: Cumulative baseline hazard rate.
    '''
    return baseline_hazard(label_E, label_T, pred_hr).cumsum()


def baseline_survival_function(label_E, label_T, pred_hr):
    '''
    :param label_E:numpy.array
           Indicators to indicate whether right censored or not.
    :param label_T:numpy.array
           Observed survival time.Negative values represent right censored.
    :param pred_hr:numpy.array
           Predicted hazard rate.
    :return:Baseline survival function.
    '''
    base_cum_hazard = baseline_cumulative_hazard(label_E, label_T, pred_hr)
    base_surv_func = np.exp(- base_cum_hazard)
    return base_surv_func


def baseline_survival_function_Breslow(y, pred_hr):
    """
    Estimate baseline survival function by Breslow Estimate.
    :param y: numpy.array
           Observed survival time.Negative values represent right censored.
    :param pred_hr:
           Predicted hazard ratio.
    :return:Estimated baseline survival function
    """
    y = np.squeeze(y)
    pred_hr = np.squeeze(pred_hr)
    T = np.abs(y)
    E = (y > 0).astype(np.int64)
    return baseline_survival_function(E, T, pred_hr)

def plot_train_curve(loss, metrics):
    '''
    Plot the loss function and concordance index during training.
    :param loss:list
           Loss function during training.
    :param metrics:list
           Concordance index during training.
    :param title:list
           The titles of two plots.
    '''
    if len(loss) != len(metrics):
        raise ValueError('The length of loss and metrics should be the same')

    if [type(loss), type(metrics)] is not [list, list]:
        loss = [l for l in loss]
        metrics = [m for m in metrics]

    plt.subplot(1,2,1)
    plt.plot(range(1, len(loss)+1), loss, color='blue', linewidth=1)
    plt.xlabel('Step')
    plt.grid(color='gray', linestyle='--')
    plt.title('Loss')

    plt.subplot(1,2,2)
    plt.plot(range(1,len(metrics)+1), metrics, color='green', linewidth=1)
    plt.xlabel('Step')
    plt.grid(color='gray', linestyle='--')
    plt.title('Concordance Index')

    plt.show()


def plot_surv_curve(surv_df, title='Survival Curve'):
    '''
    Plot survival curve.
    :param surv_df: numpy.array or pandas.DataFrame
           Survival functions of samples. The shape of surv_df should be (n, time_points).
    :param title:string
           Title of figure.
    '''

    if isinstance(surv_df, pd.DataFrame):
        plt.plot(surv_df.columns.values, np.transpose(surv_df.values))

    elif isinstance(surv_df, np.ndarray):
        plt.plot(np.array([i for i in range(1, surv_df.shape[1]+1)]), np.transpose(surv_df))

    else:
        raise TypeError('Type of survival data should be pandas.DataFrame or numpy.ndarray.')

    plt.title(title)
    plt.show()

def plot_km(data, T_col='T', E_col='E'):

    fig, ax = plt.subplots(figsize=(6,4))
    kmf = KaplanMeierFitter()
    kmf.fit(data[T_col], event_observed=data[E_col], label='KM Curve')
    kmf.survival_function_.plot(ax=ax)
    plt.ylim(0, 1.01)
    plt.xlabel('Time')
    plt.ylabel('$\hat{S}(t)$')
    plt.legend(loc='best')
    add_at_risk_counts(kmf, ax=ax)
    plt.show()
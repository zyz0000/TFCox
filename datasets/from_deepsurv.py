## coding utf-8

from pathlib import Path
import TFCox
import h5py
import pandas as pd

class DeepSurvLoader(object):
    '''
    Load real-world datasets of model DeepSurv. Please refer to

    Katzman Jared L, Shaham Uri, Cloninger Alexander, Bates Jonathan,
    Jiang Tingting, Kluger Yuval. DeepSurv: personalized treatment recommender
    system using a Cox proportional hazards deep neural network.[J].
    BMC medical research methodology,2018,18(1).

    for more information of the data.
    '''

    def __init__(self, name, split=True):
        '''
        Initialize a DataSetLoader for DeepSurv model.
        :param name: string
               The name of the dataset ot be loaded.
               Must be one of {'gbsg', 'metabric', 'support', 'whas'}
        :param split: boolean
               Whether to return the train and test data.
        '''

        self.name = name
        self.split = split
        if self.name not in ['gbsg', 'metabric', 'support', 'whas']:
            raise ValueError("The input name of the dataset should "
                             "be one of 'gbsg', 'metabric', 'support', 'whas'.")

    def load_data(self):
        PATH_ROOT = Path(TFCox.__file__).parent
        PATH_DATA = PATH_ROOT/'datasets'

        def concat(f):
            train = f['train']
            test = f['test']
            train_data = pd.concat([
                pd.DataFrame(train['x'][:]),
                pd.DataFrame(train['e'][:]),
                pd.DataFrame(train['t'][:])
            ], axis=1)
            test_data = pd.concat([
                pd.DataFrame(test['x'][:]),
                pd.DataFrame(test['e'][:]),
                pd.DataFrame(test['t'][:])
            ], axis=1)
            return train_data, test_data

        if self.name == 'metabric':
            f = h5py.File(PATH_DATA/self.name/'metabric_IHC4_clinical_train_test.h5', 'r')
            train, test = concat(f)
            new_cols = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'e', 't']
            train.columns, test.columns = new_cols, new_cols

        elif self.name == 'gbsg':
            f = h5py.File(PATH_DATA/self.name/'gbsg_cancer_train_test.h5', 'r')
            train, test = concat(f)
            new_cols = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'e', 't']
            train.columns, test.columns = new_cols, new_cols

        elif self.name == 'support':
            f = h5py.File(PATH_DATA/self.name/'support_train_test.h5', 'r')
            train, test = concat(f)
            new_cols = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7',
                        'x8', 'x9', 'x10', 'x11', 'x12', 'x13',
                        'x14', 'e', 't']
            train.columns, test.columns = new_cols, new_cols

        else:
            f = h5py.File(PATH_DATA/'whas'/'whas_train_test.h5', 'r')
            train, test = concat(f)
            new_cols = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'e', 't']
            train.columns, test.columns = new_cols, new_cols

        if not self.split:
            return pd.concat([train_data, test_data], axis=0)

        return train, test
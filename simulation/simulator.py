from math import log, exp
import numpy as np

class SimulatedData(object):
    def __init__(self, hr, average_death=5,
                 end_time=15, num_features=10,
                 num_var=2, treatment_group=False):
        '''
        A class for generating simulated survival data.
        Currently supports two types of simulated data:
            Linear: where risk is a linear combination of covariates.
            Gaussian: where risk is a gaussian form of covariates.
        :param hr:float
               'lambda_max' hazard ratio.
        :param num_features:int
               Number of features. Default value is 10.
        :param num_var:int
               Number of variables that are really used to generate simulated data.
        :param average_death:int or float.
               Average death time that is the mean of the exponential distribution(\frac{1}{\lambda}).
        :param end_time:int or float.
               Time point that represents an 'end of study'. Any death
               time greater than end_time will be censored.
        :param treatment_group:boolean
               True or False. Include an additional variable representing a binary group(male or female).
        '''

        self.hr = hr
        self.end_time = end_time
        self.average_death = average_death
        self.treatment_group = treatment_group
        self.m = int(num_features) + int(treatment_group)
        self.num_var = num_var


    def linear_hazard(self, x):
        '''
        Calculate the linear risk function.
        :param x:numpy.array
               (n,m) observation data.
        :return:numpy.array
                Risk=\sum_{k=1}^{p} kX_k, where p is parameter 'num_vars'.
        '''
        #Let the coefficients be [1,2,...,num_var,0..0].
        w = np.zeros((self.m,))
        w[0:self.num_var] = range(1, self.num_var+1)

        risk = np.matmul(x, w)
        return risk

    def gaussian_hazard(self, x, mu=0.0, sigma=1):
        '''
        Calculate the linear risk function.
        :param x:numpy.array
               (n,m) observation data.
        :param mu:float
               The mean of Gaussian function. Default value is 0.0.
        :param sigma:float
               The scale parameter of Gaussian function. Default is 1.0.
        :return:numpy.array
               Risk = \exp( \frac{\sum_{k=1} ^ {p} ( X_k - mu)^2}{2 \sigma ^2}).
        '''
        max_hr, min_hr = log(self.hr), log(1/self.hr)
        z = np.square(x-mu)
        z = np.sum(z[:,0:self.num_var], axis=-1)

        risk = max_hr*(np.exp(-z/(2*sigma**2)))
        return risk

    def generate_data(self, n_obs, method='gaussian', seed=0, gaussian_config={}, **kwargs):
        '''
        Generates a set of observations according to an exponential Cox model.
        :param n_obs:int
                The number of observations.
        :param method:string
                The type of simulated data: 'linear' or 'gaussian'.
        :param seed:int
                Random state.
        :param gaussian_config:dict
                Additional parameters for Gaussian simulation.
        :param kwargs:dict
                Dictionary arguments
        :return: dict
            {
            'x' : (n,m) numpy array of observations.
            'E' : (n) numpy array of observed time events.
            'T' : (n) numpy array of observed time intervals.
            'hr': (n) numpy array of observed true risk.
            }
        '''
        #set random state
        np.random.seed(seed)
        #patient baseline information(n,m)
        data = np.random.uniform(low=-1, high=1, size=(n_obs, self.m))
        if self.treatment_group:
            data[:,-1] = np.squeeze(np.random.randint(0,2,size=(n_obs,1)))
            print(data[:,-1])

        #each patient has a uniform death probability
        death_prob = self.average_death * np.random.uniform(0,1, size=(n_obs, 1))

        #patients hazard model
        #H(x)=log(\lambda(t) / \lambda_0(t))
        if method == 'linear':
            risk = self.linear_hazard(data)

        elif method == 'gaussian':
            risk = self.gaussian_hazard(data, **gaussian_config)

        #center the hazard ratio
        risk = risk - np.mean(risk)

        #generate time of death for each patient
        death_time = np.zeros((n_obs,1))
        for i in range(n_obs):
            if self.treatment_group and data[i,-1] == 0:
                death_time[i] = np.random.exponential(death_prob[i])
            else:
                death_time[i] = np.random.exponential(death_prob[i] / np.random.exponential(i))

        #censor anything that is past end time
        censoring = np.ones((n_obs,1))
        death_time[death_time > self.end_time] = self.end_time
        censoring[death_time == self.end_time] = 0

        #flatten arrays to vectors
        death_time = np.squeeze(death_time)
        censoring = np.squeeze(censoring)

        dataset = {
            'x' : data.astype(np.float32),
            'E' : censoring.astype(np.float32),
            'T' : death_time.astype(np.float32),
            'hr': risk.astype(np.float32)
        }
        return dataset
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 13:43:01 2015
@author: rfsantacruz
Parametric Models module:
Content:
    - Parametric Methods
        - Distributions
            - Bernoulli
            - Binomial
            - Multinomial
            - Gaussian
            - Studnet T
        - Parameter Estimation
            - Maximum Likelihood
            - Bayesian (using conjugate prior)
"""


import numpy as np
from scipy import special as sp
from scipy import stats as st
from enum import Enum



class Estimator(Enum):
    MLE = 1;
    BAYESIAN = 2;

""" Distribution of x = 1, given that the data set has size N """
class Bernoulli:

    """given observations and the method of parameter estimation"""
    def __init__(self, obs=None, a=0, b=0, est=Estimator.MLE):
        
        self._est = est;                        
        self._mu = 0;
        self._m = 0;        
        self._l = 0;  
        self._beta_cp = None; 
        
        if(est == Estimator.MLE):
            self._N = obs.size;
            self._mu = np.sum(obs)/self._N;        
            
            
        if(est == Estimator.BAYESIAN):        
            self._N = 0;
            self._m = 0;            
            self._l = 0;
            self._beta_cp = Beta(a , b);
        
    
    """ probability of outcome given pre computed mu """    
    def prob(self, X):
        
        prob = 0.0;
        if(self._est == Estimator.MLE):
            prob = np.power(self._mu, X) * np.power(1 - self._mu, 1 - X);
        else:
            p_1 = self._beta_cp.expec();
            prob = np.power(p_1, X) * np.power(1 - p_1, 1 - X);

        return prob;            
    
    """expectation"""    
    def expec(self):
        if(self._est == Estimator.MLE):
            return self._mu;
        else: #bayesian
            return self._beta_cp.expec();
        
    """variance"""
    def var(self):
        if(self._est == Estimator.MLE):
            return self._mu * (1 - self._mu);
        else: #bayesian
            return self._beta_cp.var();
            
    """update just for bayesian"""
    def update(self, obs):
        if(self._est == Estimator.BAYESIAN):                        
            
            #update beta conjugate prior given new observations
            m = np.sum(obs); l = obs.size - m;                        
            self._beta_cp.update(m, l); 
            
            #update attributes            
            self._N = self._N + m + l;              
            self._m = self._m + m;
            self._l = self._l + l;
            
        else: 
             raise NotImplementedError;
    

        
    """ to string """
    def __str__(self):
        return 'Bernoulli distribution:\n' \
        'Outcomes: Binary [0,1]\n' \
        'Num of observations: {}\n' \
        'Estimator: {}\n' \
        'Expec: {:.2f}\n' \
        'Var: {:.2f}\n' \
        'Probability of (0,1) = {}'.format(self._N, self._est, self.expec(), self.var(), self.prob(np.array([0,1])));
            

""" Distribution of the number m of observations of x = 1, given N Observations """
class Binomial:
    
    """given observations and the method of parameter estimation"""
    def __init__(self, obs=None, a=0, b=0, est=Estimator.MLE):
        
        self._est = est;            
        
        
        if(est == Estimator.MLE):
            self._N = obs.size;            
            self._ben = Bernoulli(obs, est=Estimator.MLE);

        if(est == Estimator.BAYESIAN):
            self._N = 0;
            self._ben = Bernoulli(obs, a, b, est=Estimator.BAYESIAN);
    
    """ probability of have m x=1, given observations """    
    def prob(self, M):
 
        # Probability of have m x=1       
        mu = self._ben.prob(1);                    
        prob =  np.power(mu, M) * np.power(1 - mu, self._N - M);            
        # Combinatorial of way to have m's x = 1
        comb = sp.factorial(self._N - M) * sp.factorial(M);
        comb = sp.factorial(self._N)/comb;
        return comb * prob;           
        
    """expectation"""    
    def expec(self):
        return self._N * self._ben.expec();
    
    """variance"""
    def var(self):
        ex = self._ben.expec();
        return self._N * ex * (1 - ex);
    
    """update just for bayesian"""
    def update(self, obs):        
        self._ben.update(obs);        
        self._N = self._N + obs.size;                      
        
            
    """ to string """
    def __str__(self):
        return 'Binomial distribution:\n' \
        'Outcomes: Discrete integer\n' \
        'Num of observations: {}\n' \
        'Estimator: {}\n' \
        'Expec: {:.2f}\n' \
        'Var: {:.2f}'.format(self._N, self._est, self.expec(), self.var());
    

""" Beta prior conjugate distributions is the conjugate prior of bernoulli and binomial distribution """
class Beta:
    
    """integer hyperparameters define the form of the distribution"""
    def __init__(self, a, b ):       
        self._a = a;
        self._b = b;        
    
    """ prior probability of mu given hyperparameters """
    def prob(self, mu):
        prob =  np.power(mu, self._a - 1) * np.power(1 - mu, self._b - 1);    
        norm = sp.gamma(self._a + self._b)/(sp.gamma(self._a) * sp.gamma(self._b));
        return norm * prob;         
        
    """expectation"""    
    def expec(self):
        return self._a / (self._a + self._b);
    
    """variance"""
    def var(self):
        return (self._a * self._b) / (pow((self._a + self._b),2) * (self._a + self._b + 1));
            
    def update(self, m, l):
        self._a = self._a + m;
        self._b = self._b + l;        
        
    """ to string """
    def __str__(self):
        return 'Beta conjugate prior distribution:\n' \
        'Outcomes: Continuous\n' \
        'a: {}\n' \
        'b: {}\n' \
        'Expec: {:.2f}\n' \
        'Var: {:.2f}'.format(self._a, self._b, self.expec(), self.var());


""" Multinomial distribution of discrete variables """
class Multinomial:
    
    """given observations and the method of parameter estimation"""
    def __init__(self, obs, K, alphas=None, est=Estimator.MLE):
        
        self._est = est;               
        self._N = obs.size;
        self._K = K;
        
        if(est == Estimator.MLE):
            self._m = np.zeros(self._K);            
            for k in range(0,K):
                self._m[k] =  np.sum(np.equal(obs, k));               
                                    
            self._mu = self._m / self._N;
            
        if(est == Estimator.BAYESIAN):            
            self._m = np.zeros(self._K);            
            for k in range(0,K):
                self._m[k] =  np.sum(np.equal(obs, k));
             
            self._dirichlet_cp = Dirichlet(alphas + self._m);
                                    
    """ probability of have mk_1, mk_2, mk_k, given observations """    
    def prob(self, M):
        if(self._est == Estimator.MLE):
            prob = np.prod(np.power(self._mu,self._m));
            
        if(self._est == Estimator.BAYESIAN):            
            mu = self._dirichlet_cp.expec(np.arange(0,self._K));
            prob = np.prod(np.power(mu, self._m));
        
        comb = sp.factorial(self._N) / np.prod(sp.factorial(self._m));
        return comb * prob;           
        
    """expectation"""    
    def expec(self, k):
        if(self._est == Estimator.MLE):
            return self._N * self._mu[k];    
        else:
            return self._N * self._dirichlet_cp.expec(k); 
    
    """variance"""
    def var(self, k):
        if(self._est == Estimator.MLE):
            return self._N * self._mu[k] * (1-self._mu[k]);
        else:
           mu = self._dirichlet_cp.expec(k);
           return self._N * mu * (1 - mu); 
    
    """update just for bayesian"""
    def update(self, obs):
        
        m_new = np.zeros(self._K);            
        for k in range(0,self._K):
            m_new[k] =  np.sum(np.equal(obs, k));                
        
        
        self._dirichlet_cp.update(m_new);
        self._m = self._m + m_new;
        self._N = self._N + obs.size;        
        
    """ to string """
    def __str__(self):
        return 'Multinomial distribution:\n' \
        'Outcomes: Discrete integer 0...{:d}\n' \
        'Num of observations: {}\n' \
        'Estimator: {}\n' \
        'Expec: {:.2f}\n' \
        'Var: {:.2f}'.format(self._K, self._N, self._est, 0, 0);

""" Dirichlet deistributions as a prior conjugate of Multinomial distributions"""
class Dirichlet:
    
    """integer hyperparameters define the form of the distribution"""
    def __init__(self, alphas):       
        self._alphas = alphas;
        self._K = alphas.size;
        self._norm = np.sum(alphas);        
    
    """ prior probability of mu given hyperparameters """
    def prob(self, mu):
        prob = np.prod(np.power(mu, self._alphas - 1));
        norm = sp.gamma(self._norm)/(np.prod(sp.gamma(self._alphas)));
        return norm * prob;         
        
    """expectation"""    
    def expec(self, k):
        return self._alphas[k] / self._norm ;
    
    """variance"""
    def var(self, k):
        return (self._alphas[k] * (self._norm - self._alphas[k])) / (pow(self._norm,2)*(self._norm + 1));
            
    def update(self, alphas_obs):        
        self._alphas = self._alphas + alphas_obs;         
        self._norm =  np.sum(self._alphas);
        
    """ to string """
    def __str__(self):
        return 'Dirichlet prior conjugate distribution:\n' \
        'Outcomes: Discrete integer 0...{:d}\n' \
        'Expec: {:.2f}\n' \
        'Var: {:.2f}'.format(self._K, 0, 0);
        

""" Gaussian univariate distribution with help of scipy in the equations"""        
class Gaussian:
    
    def __init__(self, obs, est=Estimator.MLE):

        self._N = obs.size;        
        self._est = est;        
        
        if(est == Estimator.MLE):
            self._mu = np.sum(obs)/ self._N;
            self._var = np.sum(np.power(obs - self._mu,2)) / self._N;
        else:
            raise Exception;
                        
    def prob(self, X):
        return st.norm.pdf(X, self._mu, self._var);
        
    """expectation"""    
    def expec(self):
        return self._mu;
    
    """variance"""
    def var(self):
        return self._var;
            
    def update(self):        
        raise Exception;
        
    """ to string """
    def __str__(self):
        return 'Univariate Gaussian distribution:\n' \
        'Outcomes: Continuous values\n' \
        'Observations: {:d}\n'
        'Expec: {:.2f}\n' \
        'Var: {:.2f}'.format(self._N, self.expec(), self.var());
    

""" Multivariate Gaussian distribution """
class MultGaussian:
    
    def __init__(self, obs, est=Estimator.MLE):
        (self._N, self._D) = obs.shape;        
        self._est = est;        
        
        if(est == Estimator.MLE):
            self._mu_vec = np.sum(obs, axis=0) / self._N;
            self._cov = np.cov(obs);
        else:
            raise Exception;
    
    def prob(self, X):
        return st.multivariate_normal.pdf(X, mean=self._mu_vec, cov=self._cov);
        
    """expectation"""    
    def expec(self):
        return self._mu_vec;
    
    """variance"""
    def var(self):
        return self._cov;
            
    def update(self):        
        raise Exception;
        
    """ to string """
    def __str__(self):
        return 'Multivariate Gaussian distribution:\n' \
        'Outcomes: Continuous values\n' \
        'Observations: matrix {:d}{:d}\n' \
        'Expec: {:.2f}\n' \
        'Var: {:.2f}'.format(self._N, self._D, self.expec(), self.var());
        
        
"""Gaussian Gama distribution, prior conjugate for univarate gaussian distribution with unknown mean and covariance"""
class Gaussian_Gama:
    
    def __init__(self, mu, k, alpha, beta):
        self._mu = mu;
        self._k = k;
        self._alpha = alpha;
        self._beta = beta;
    
    def prob(self, mu, prec):                
        return st.norm.pdf(mu, loc = self._mu, scale = np.sqrt(1.0/self._k*prec))  \
        * st.gamma.pdf(prec, self._alpha, scale = np.sqrt(1.0/self._beta));       
        
        
    """expectation"""    
    def expec(self):
        return np.array([self._mu, self._alpha*pow(self._b,-1)]);
    
    """variance"""
    def var(self):
        return np.array([self._beta/(self._k*(self._alpha - 1)), self._alpha*pow(self._beta,-2)]);
            
    def update(self, n, mean, sq_dif):
        self._mu = (self._mu * self._k + n*mean)/(self._k + n);
        self._k = self._k + n;
        self._alpha = self._alpha + (n/2.0);
        self._beta = self._beta + 0.5 * sq_dif + ((self._k*n*pow(mean - self._mu,2)) / (2*(self._k + n)));
        
    """ to string """
    def __str__(self):
        return 'Gaussian Gama distribution:\n' \
        'Outcomes: mean and variance\n' \
        'Expec: {}\n' \
        'Var: {}'.format(self._N, self._D, self.expec(), self.var());        


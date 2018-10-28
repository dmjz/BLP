import math
import pandas as pd
import numpy as np
from scipy import optimize


# Prior portfolio for a model:
# ..assetClasses = tuple of strings
# ..weights = list of doubles, same length as assetClasses
# ..riskAversion = double
# ..covMatrix = covariance matrix of expected excess returns for assetClasses
class Portfolio:
    def __init__(self, 
        assetClasses=('US Equity', 'Foreign EQ', 'Emerging EQ'),
        assetWeights=[0.33,0.33,0.34],
        riskAversion=2.5,
        covMatrix=None
    ):
        self.assetClasses = assetClasses
        self.weights = np.asarray(assetWeights)
        self.riskAversion = riskAversion
        self.covMatrix = covMatrix
        if covMatrix is None:
            self.covMatrix = pd.DataFrame(
                dict(zip(self.assetClasses, [[0,0]]*len(self.assetClasses)))
            ).cov()
        self.returns = self.computeReturns()
        
    def computeReturns(self):
        # Converting from 1D array to 2D column vector
        return (
            np.atleast_2d(
                self.riskAversion * np.dot(self.covMatrix, self.weights)
            ).T
        )
       
        
# Parameters for a model:
# ..priorPortfolio = Portfolio object
# ..tau = double
# ..tauv = double
# ..P = numpy matrix kXn, where k = # of views, n = # of asset classes
# ..Q = numpy matrix kX1, where k = # of views
class Parameters:
    def __init__(self, priorPortfolio, tau, tauv, P, Q):
        self.prior = priorPortfolio
        self.Pi = priorPortfolio.returns
        self.Sigma = priorPortfolio.covMatrix.values #pd DataFrame -> np array
        self.tau = tau
        self.tauv = tauv
        self.P = P
        self.Q = Q
        self.Omega = self.computeOmega()
        self.PiHat = self.computePostReturns()
        self.SigmaP = self.computePostVariance()
        
    def computeOmega(self):
        #Omega = tauv*P*Sigma*P_T
        return self.tauv*(np.dot(np.dot(self.P, self.Sigma), self.P.T))
    
    def computePostReturns(self):
        #PiHat = Pi + (tau*Sigma*P_T) * [(P*Sigma*P_T)+Omega]^(-1) * [Q - P*Pi]
        #      = Pi + a * b^(-1) * c
        a = self.tau*np.dot(self.Sigma, self.P.T)
        b = np.dot(np.dot(self.P, self.Sigma), self.P.T) + self.Omega
        c = self.Q - np.dot(self.P, self.Pi)
        return self.Pi + np.dot(np.dot(a, np.linalg.inv(b)), c)
        
    def computePostVariance(self):
        #SigmaP = Sigma + (tau*Sigma) 
        #         - (tau*Sigma*P.T) * [P*tau*Sigma*P.T + Omega]^(-1) * P*tau*Sigma
        #       = u - a * b^(-1) * c
        u = self.Sigma + self.tau*self.Sigma
        a = self.tau*np.dot(self.Sigma, self.P.T)
        b = self.tau*np.dot(np.dot(self.P, self.Sigma), self.P.T) + self.Omega
        c = self.tau*np.dot(self.P, self.Sigma)
        return u - np.dot(np.dot(a, np.linalg.inv(b)), c)
        
    def postDistribution(self):
        return (np.ndarray.flatten(self.PiHat), self.SigmaP)
        

# Find weights which maximize Sharpe ratio
#..parameters = Parameters object
class Optimizer:
    def __init__(self, parameters):
        self.parameters = parameters
        self.result = self.optimizeWeights()
        self.success = self.result.success
        if self.result.success:
            self.optimalWeights = self.result.x
            self.optimalReturn = self.expectedReturn(self.optimalWeights)
            self.optimalSd = self.sd(self.optimalWeights)
            self.optimalSharpe = self.sharpe(self.optimalWeights)
        else:
            print('In Optimizer: there was a problem finding the optimal weights')
            print('Return message from optimize.minimize:')
            print(self.result.message)
        
    def expectedReturn(self, weights):
        return np.asscalar(
            np.dot(np.atleast_2d(weights), self.parameters.PiHat)
        ) 
        
    def sd(self, weights):
        w = np.atleast_2d(weights)
        weightedCov = np.dot(np.dot(w, self.parameters.SigmaP), w.T)
        return math.sqrt(np.sum(weightedCov))
        
    def sharpe(self, weights):
        return self.expectedReturn(weights)/self.sd(weights)
        
    def optimizeWeights(self):
        optResult = optimize.minimize(
            fun=lambda x: -1*self.sharpe(np.atleast_2d(x)),
            x0=np.ndarray.flatten(self.parameters.prior.weights),
            method='SLSQP',
            constraints={
                'fun': lambda x: 1 - np.sum(x),
                'type': 'eq'
            }
        )
        return optResult
    
    def optimalData(self):
        if self.success:
            return (
                list(self.optimalWeights),
                self.optimalReturn,
                self.optimalSd,
                self.optimalSharpe
            )


# Create an optimal portfolio using the Black-Litterman model
# ..prior = a Portfolio object representing the prior return distribution
# ......Or pass assetClasses, weights, riskAversion, and covMatrix
# ......to the class method Model.fromPriorData. See Portfolio definition
# ......above
# ..tau = a scalar constant
# ..tauv = a scalar constant
# ..P = numpy matrix kXn, where k = # of views, n = # of asset classes
# ..Q = numpy matrix kX1, where k = # of views
# ..identifier = number or string to identify the model, default -1
class Model:
    def __init__(self, prior, tau, tauv, P, Q, identifier=-1):
        self.identifier = identifier
        self.prior = prior
        self.assetClasses = prior.assetClasses
        self.tau = tau
        self.tauv = tauv
        self.P = P
        self.Q = Q
        self.parameters = Parameters(prior, tau, tauv, P, Q)
        self.optimizer = Optimizer(self.parameters)
        if self.optimizer.success:
            self.weights, self.returns, self.sd, self.sharpe = self.optimizer.optimalData()
            self.optimalPortfolio = dict(zip(
                ('weights', 'returns', 'sd', 'sharpe'),
                self.optimizer.optimalData()
            ))
            self.framer = ModelFramer(self, self.identifier)
            self.df = self.framer.df
        else:
            print('In Model: there was a problem finding the optimal weights')
            print('Return message from optimize.minimize:')
            print(self.optimizer.result.message)
    
    @classmethod
    def fromPriorData(cls,
        assetClasses, assetWeights, riskAversion, covMatrix,
        tau, tauv, P, Q,
        identifier=-1
    ):
        prior = Portfolio(assetClasses, assetWeights, riskAversion, covMatrix)
        return cls(prior, tau, tauv, P, Q, identifier)
            
    def didGeneratePortfolio(self):
        return self.optimizer.success
            
    def posteriorDistribution(self):
        return self.parameters.postDistribution()
        
    def listPriorReturns(self):
        return list(np.ndarray.flatten(self.prior.returns))
    
    def listPriorWeights(self):
        return list(self.prior.weights)
        
        
# Build a pandas DataFrame out of a Model, for printing to csv
# ..model = Model object
# ..identifier = number or string to identify the model (default -1)
class ModelFramer:
    def __init__(self, model, identifier=-1):
        self.model = model
        self.identifier = identifier
        self.prior = self.makePrior()
        self.head = self.makeHead()
        self.views = self.makeViews()
        self.dist = self.makeDist()
        self.optimal = self.makeOptimal()
        self.df = pd.concat((self.head, self.views, self.dist, self.optimal))
        
    # Note: expects a numpy array for covMatrixValues
    def covDf(self, covMatrixValues, label):
        df = pd.concat((
                pd.DataFrame.from_dict({
                    label: self.model.assetClasses
                }, orient='index'),
                pd.DataFrame(covMatrixValues)
            ))
        df.index = [label] + self.model.assetClasses
        return df
        
    def makePrior(self):
        return pd.concat((
            pd.DataFrame.from_dict({
                'Asset classes': self.model.assetClasses,
                'Prior weights': self.model.listPriorWeights(),
                'Risk aversion': [self.model.prior.riskAversion],
                'Prior returns': self.model.listPriorReturns()
            }, orient='index'),
            self.covDf(self.model.prior.covMatrix.values, 'Prior covariance matrix')
        ))
        
    def makeHead(self):
        return pd.DataFrame.from_dict({
            'Model Identifier': [self.identifier],
            'Prior weights': list(self.model.prior.weights),
            'Parameters': [
                'Risk aversion', self.model.prior.riskAversion,
                'Tau', self.model.parameters.tau,
                'Tauv', self.model.parameters.tauv
                ],
            }, orient='index')
    
    def makeViews(self):
        assetClasses = self.model.assetClasses
        dfP = pd.concat((
            pd.DataFrame.from_dict({'P': assetClasses}, orient='index'),
            pd.DataFrame(self.model.parameters.P)
        ))
        dfQ = pd.concat((
            pd.DataFrame.from_dict({'Q': []}, orient='index'),
            pd.DataFrame(self.model.parameters.Q)
        ))
        df = pd.concat((dfP, dfQ))
        df.index = (
            ['P'] + ['View ' + str(ind) for ind in dfP.index[1:]] +
            ['Q'] + ['View ' + str(ind) for ind in dfQ.index[1:]]
        )
        return df
        
    def makeDist(self):
        postReturns, postVars = self.model.posteriorDistribution()
        return pd.concat((
            pd.DataFrame.from_dict({
                'Posterior expected returns': postReturns
                }, orient='index'),
            self.covDf(postVars, 'Posterior covariance matrix')
        ))
        
    def makeOptimal(self):
        op = self.model.optimalPortfolio
        return pd.DataFrame.from_dict({
            'Optimal weights': op['weights'],
            'Optimal portfolio metrics': [
                'Expected return', op['returns'],
                'Sd', op['sd'],
                'Sharpe', op['sharpe']
                ]
        }, orient='index')


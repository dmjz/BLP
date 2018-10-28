# BLP

## A portfolio optimizer based on the Black-Litterman model

(Note: "returns" always refers to returns in excess of the risk-free rate)

The Black-Litterman model takes prior return data for a collection of asset classes, and a manager's views on future returns in those asset classes, and estimates the future distribution of returns. This can be used to determine the optimal weights for a portfolio in those asset classes.

BLP.py provides the Model class to implement the B-L model and determine the optimal weights in a collection of asset classes to maximize the Sharpe ratio of the portfolio.

In example_script.py, we use return data from example_returndata.csv to define three models, with the same asset classes and prior weights but different parameters, and print the model information and computed optimal weights to example_output.csv.


#### Parameters to define a Model:
1. A list of asset classes
2. A list of prior weights in the asset classes
3. The covariance matrix of the excess returns of the asset classes
4. A scalar risk aversion parameter
5. tau: a scalar parameter that weights the covariance matrix
6. tauv: a scalar parameter that weights the manager's views
7. P: a numpy matrix expressing the types of the manager's views. See explanation below. P should have dimensions KxN, where K is the number of views and N is the number of asset classes
8. Q: a numpy matrix expressing the quantities of the manager's views. See explanation below. Q should have dimensions Kx1, where K is the number of views
9. An optional identifier, which can be a string or a scalar

#### Example parameters:
For the first model in example_script.py, we use:
```python
model_one = BLP.Model.fromPriorData(
    assetClasses=['US Equity', 'Foreign EQ', 'Emerging EQ'], 
    assetWeights=[0.5, 0.4, 0.1], 
    riskAversion=3, 
    covMatrix=covMatrix,
    tau=0.1,
    tauv=0.1, 
    P=np.asarray(
      [[1,0,0], 
       [0,1,-1]]
       ),
    Q=np.asarray(
      [[0.015],
       [0.03]]
       ),
    identifier=1
)
```

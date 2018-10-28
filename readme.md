# BLP

## A portfolio optimizer based on the Black-Litterman model

(Note: "returns" always refers to returns in excess of the risk-free rate)

The Black-Litterman model takes prior return data for a collection of asset classes, and a manager's views on future returns in those asset classes, and estimates the future distribution of returns. This can be used to determine the optimal weights for a portfolio in those asset classes.

BLP.py provides the Model class to implement the B-L model and determine the optimal weights in a collection of asset classes to maximize the Sharpe ratio of the portfolio.

In example_script.py, we use return data from example_returndata.csv to define three models, with the same asset classes and prior weights but different parameters, and print the model information and computed optimal weights to example_output.csv.

import csv
from collections import OrderedDict
import pandas as pd
import numpy as np
import BLP
        
        
def readDataToFrame(filename, keepColumns):
    print('Reading data from "' + filename + '"...')
    with open(filename) as csvfile:
        
        readCols = {colname: [] for colname in keepColumns}
        reader = csv.DictReader(csvfile)
        for row in reader:
            for colName in readCols:
                if colName in row:
                    readCols[colName].append(row[colName])
        df = pd.DataFrame(readCols)
        
        print('Done.')
        return df.apply(pd.to_numeric)
        
def writeResults(filename, models):
    print('Writing output to "' + filename + '"...')
    writeDf = pd.concat([models[0].framer.prior] + [m.df for m in models])
    writeDf.to_csv(filename, header=False)
    print('Done.')
    

# Setup:
assetInfo = {'US Equity': 0.5, 'Foreign EQ': 0.4, 'Emerging EQ': 0.1}
assetClasses = list(assetInfo.keys())
assetWeights = list(assetInfo.values())  

data = readDataToFrame('example_returndata.csv', keepColumns=assetClasses)
covMatrix = data.cov()

# Models:
print('Computing models...')
model_one = BLP.Model.fromPriorData(
    assetClasses, 
    assetWeights, 
    riskAversion=3, 
    covMatrix=covMatrix,
    tau=0.1,
    tauv=0.1, 
    P=np.asarray([[1,0,0], [0,1,-1]]),
    Q=np.asarray([[0.015],[0.03]]),
    identifier=1
)
model_two = BLP.Model.fromPriorData(
    assetClasses, 
    assetWeights, 
    riskAversion=3, 
    covMatrix=covMatrix,
    tau=0.1,
    tauv=0.01, 
    P=np.asarray([[1,0,0], [0,1,-1]]),
    Q=np.asarray([[0.015],[0.03]]),
    identifier=2
)
model_three = BLP.Model.fromPriorData(
    assetClasses, 
    assetWeights, 
    riskAversion=3, 
    covMatrix=covMatrix,
    tau=0.1,
    tauv=0.01, 
    P=np.asarray([[1,-1,0], [0,0,1]]),
    Q=np.asarray([[0.02],[0.015]]),
    identifier=3
)
models = (model_one, model_two, model_three)
print('Done.')

# Write results:
writeResults('new_example_output.csv', models)
        

import numpy as np
import pandas as pd


def cleanData(data):
    data["score"] = data["score"].apply(lambda x:  str(x).replace(",","."))
    data = data.drop(data[data["user score"] == "tbd"].index)
    data["user score"] = data["user score"].apply(lambda x:  str(x).replace(",","."))
    data["score"] = data["score"].astype(np.float64)
    data["user score"] = data["user score"].astype(np.float64)
    data["critics"] = data["critics"].astype(np.float64)
    data["users"] = data["users"].astype(np.float64)
    data["score"] = data["score"].apply(lambda x: x/10)
    return data

def load_data_csv(path,x_colum,y_colum):
    data = pd.read_csv(path)
    data = cleanData(data)
    X = data[x_colum].to_numpy()
    y = data[y_colum].to_numpy()
    return X, y

def zscore_normalize_features(X):
    """
    computes  X, zcore normalized by column

    Args:
      X (ndarray (m,n))     : input data, m examples, n features

    Returns:
      X_norm (ndarray (m,n)): input normalized by column
      mu (ndarray (n,))     : mean of each feature
      sigma (ndarray (n,))  : standard deviation of each feature
    """
    mu = np.mean(X, axis=1)
    sigma = np.std(X, axis=1)
    x1 = (X[0]-mu[0])/sigma[0]
    x2 = (X[1]-mu[1])/sigma[1]
    x3 = (X[2]-mu[2])/sigma[2]
    X_norm = np.append([x1,x2],[x3],axis=0)

    X_norm = np.transpose(X_norm)
   
    return X_norm, mu, sigma

def load_data_csv_multi(path,x1_colum,x2_colum,x3_colum,y_colum):
    data = pd.read_csv(path)
    data = cleanData(data)
    x1 = data[x1_colum].to_numpy()
    x2 = data[x2_colum].to_numpy()
    x3 = data[x3_colum].to_numpy()
    X = np.array([x1, x2, x3])
    #X = X.T
    y = data[y_colum].to_numpy()
    return X, y

def pasar_0_1(param):
    if(param<7):
        param = 0
    else:
        param = 1
    return param

## 0 Malo, 1 Regular, 2 Notable, 3 Sobresaliente, 4 Must Play.
## 0 Malo, 1 Bueno
def load_data_csv_multi_logistic(path,x1_colum,x2_colum,x3_colum,y_colum):
    X,y = load_data_csv_multi(path,x1_colum,x2_colum,x3_colum,y_colum)
    y[y<7]=0
    y[y>=7]=1
    #TODO convertir la a clases 0,1.
    return X,y
        
def GetNumGradientsSuccess(w1,w1Sol,b1,b1Sol):
    iterator = 0
    for i in range(len(w1)): 
        if np.isclose(w1[i],w1Sol[i]):
                iterator += 1
    if np.isclose(b1,b1Sol):
        iterator += 1
    return iterator
    
        
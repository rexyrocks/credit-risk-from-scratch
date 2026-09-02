import pandas as pd
import numpy as np
def load_data(path):
    df = pd.read_csv(path)
    return df
def cleandata(df):
    df = df.drop(columns = ["Unnamed: 0"])
    df["MonthlyIncome"] = df["MonthlyIncome"].fillna(df["MonthlyIncome"].median())
    df["NumberOfDependents"] = df["NumberOfDependents"].fillna(df["NumberOfDependents"].mode()[0])
    return df
def scalefeature(X):
    mean = X.mean(axis = 0)
    std = X.std(axis =0)
    X_sclaed = (X-mean)/std
    return X_sclaed,mean,std
def train_test_split(X,y,test_size = 0.2 , seed = 42):
    np.random.seed(seed)
    n = X.shape[0]
    indices = np.random.permutation(n)
    test_count = int(n*test_size)
    test_idx = indices[:test_count]
    train_idx = indices[test_count:]
    return X[train_idx] , X[test_idx], y[train_idx], y[test_idx]
if __name__ == "__main__":
    df = load_data("data/cs-training.csv")
    print(df.shape)
    df = cleandata(df)
    print(df.isnull().sum())
    
    y = df["SeriousDlqin2yrs"].values        
    X = df.drop(columns=["SeriousDlqin2yrs"]).values
    
    X_scaled, mean, std = scalefeature(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y)
    
    print("X_train:", X_train.shape)         
    print("X_test:", X_test.shape)
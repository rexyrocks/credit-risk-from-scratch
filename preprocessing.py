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
if __name__ == "__main__":
    df = load_data("data/cs-training.csv")
    print(df.shape)
    df = cleandata(df)
    print(df.isnull().sum())

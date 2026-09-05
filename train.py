import numpy as np
from preprocessing import load_data,cleandata,scalefeature,train_test_split
from logistic_regression import logisitic_regressionScratch

if __name__ == "__main__":
    df = load_data("/Users/kunalchoudhary/Desktop/kunal coding documents/project/data/cs-training.csv")
    df = cleandata(df)
    y = df["SeriousDlqin2yrs"].values
    x = df.drop(columns=["SeriousDlqin2yrs"]).values
    X_sclaed ,mean ,std = scalefeature(x)
    X_train , X_test , Y_train , Y_test = train_test_split(X_sclaed,y)
    model = logisitic_regressionScratch(learning_rate= 0.01 , n_iterations= 1000)
    model.fit(X_train,Y_train,class_weight="balanced")
    train_preds = model.predict(X_train)
    train_accuracy = np.mean(train_preds == Y_train)
    print("Train accuracy:", train_accuracy)
    print("Default rate:", np.mean(Y_train))
    probs = model.predict_proba(X_train)
    print("Min prob:", probs.min())
    print("Max prob:", probs.max())
    print("Mean prob:", probs.mean())
    train_preds_lower = model.predict(X_train, threshold=0.3)
    print("Predicted defaults (threshold 0.3):", np.sum(train_preds_lower))

    train_preds_lower2 = model.predict(X_train, threshold=0.2)
    print("Predicted defaults (threshold 0.2):", np.sum(train_preds_lower2))
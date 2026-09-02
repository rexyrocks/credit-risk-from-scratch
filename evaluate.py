import numpy as np
from logistic_regression import logisitic_regressionScratch
from preprocessing import cleandata, load_data, train_test_split, scalefeature

def confusematrix_compenents(y_true,y_pred):
    TP = np.sum((y_pred == 1) & (y_true == 1))
    TN = np.sum((y_pred == 0) & (y_true == 0))
    FP = np.sum((y_pred == 1) & (y_true == 0))
    FN = np.sum((y_pred == 0) & (y_true == 1))
    return TP ,TN , FP ,FN
def compute_metrics(TP,TN,FP,FN):
    accuracy = (TP +TN) / (TP+ TN +FN +FP)
    precision = TP / (TP+FP) if (TP+FP)>0 else 0
    recall = (TP)/(TP+FN) if (TP+FN) > 0 else 0
    f1 = 2 *(precision*recall)/ (precision+recall) if (precision+recall) > 0 else 0
    return accuracy,precision,recall,f1
if __name__ == "__main__":
    df = load_data("data/cs-training.csv")
    df = cleandata(df)
    y = df["SeriousDlqin2yrs"].values
    x = df.drop(columns=["SeriousDlqin2yrs"]).values
    X_scaled ,mean ,std = scalefeature(x)
    X_train , X_test , Y_train , Y_test = train_test_split(X_scaled,y)
    model = logisitic_regressionScratch(learning_rate=0.01 , n_iterations= 1000)
    model.fit(X_train,Y_train)
    test_preds = model.predict(X_test)
    TP , TN , FP , FN = confusematrix_compenents(Y_test,test_preds)
    accuracy , preicision , recall ,f1 = compute_metrics(TP,TN,FP,FN)
    print("TP " , TP)
    print("TN " , TN)
    print("FP " , FP)
    print("FN " , FN)
    print("Accuracy " , accuracy)
    print("recall " , recall)
    print("f1 " , f1)
    print("precision " ,preicision)
    print("\n--- Threshold Tuning ---")
    for t in [0.5, 0.4, 0.3, 0.2, 0.1, 0.05]:
            preds_t = model.predict(X_test, threshold=t)
            TP_t, TN_t, FP_t, FN_t = confusematrix_compenents(Y_test, preds_t)
            acc_t, prec_t, rec_t, f1_t = compute_metrics(TP_t, TN_t, FP_t, FN_t)
            print(f"Threshold {t}: Precision={prec_t:.3f}, Recall={rec_t:.3f}, F1={f1_t:.3f}")
    for t in [0.20, 0.18, 0.16, 0.14, 0.12, 0.10]:
           preds_t = model.predict(X_test, threshold=t)
           TP_t, TN_t, FP_t, FN_t = confusematrix_compenents(Y_test, preds_t)
           acc_t, prec_t, rec_t, f1_t = compute_metrics(TP_t, TN_t, FP_t, FN_t)
           print(f"Threshold {t}: Precision={prec_t:.3f}, Recall={rec_t:.3f}, F1={f1_t:.3f}")
    from sklearn.linear_model import LogisticRegression
    sklearn_model = LogisticRegression()
    sklearn_model.fit(X_train, Y_train)
    sklearn_preds = sklearn_model.predict(X_test)

    TP_sk, TN_sk, FP_sk, FN_sk = confusematrix_compenents(Y_test, sklearn_preds)
    acc_sk, prec_sk, rec_sk, f1_sk = compute_metrics(TP_sk, TN_sk, FP_sk, FN_sk)

    print("\n--- Sklearn Comparison (threshold 0.5 default) ---")
    print(f"Sklearn: Precision={prec_sk:.3f}, Recall={rec_sk:.3f}, F1={f1_sk:.3f}")

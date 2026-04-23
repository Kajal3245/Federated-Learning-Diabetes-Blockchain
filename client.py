from model import get_model
from sklearn.utils import resample
import numpy as np

def balance_data(X, y):
    X = np.array(X)
    y = np.array(y)

    X_0 = X[y == 0]
    X_1 = X[y == 1]

    # Upsample minority class
    if len(X_1) < len(X_0):
        X_1_upsampled = resample(X_1, replace=True, n_samples=len(X_0), random_state=42)
        y_1 = np.ones(len(X_1_upsampled))
        X_bal = np.vstack((X_0, X_1_upsampled))
        y_bal = np.hstack((np.zeros(len(X_0)), y_1))
    else:
        X_bal, y_bal = X, y

    return X_bal, y_bal


def train_client(X, y):
    X_bal, y_bal = balance_data(X, y)

    model = get_model()
    model.fit(X_bal, y_bal)

    return model.coef_, model.intercept_
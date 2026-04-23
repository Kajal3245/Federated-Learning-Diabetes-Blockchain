import numpy as np
from model import get_model

def aggregate(client_weights, client_sizes):
    total = sum(client_sizes)

    avg_coef = sum(w * (size / total) for (w, _), size in zip(client_weights, client_sizes))
    avg_intercept = sum(b * (size / total) for (_, b), size in zip(client_weights, client_sizes))

    model = get_model()
    model.coef_ = avg_coef
    model.intercept_ = avg_intercept

    return model
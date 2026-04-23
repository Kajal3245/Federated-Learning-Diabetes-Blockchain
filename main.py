from utils import load_data
from client import train_client
from server import aggregate
from model import get_model

import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

# Load data
X_train, X_test, y_train, y_test = load_data()

# Convert to numpy
X_train = X_train.values
y_train = y_train.values
X_test = X_test.values
y_test = y_test.values

# Create clients (simulate 3 hospitals)
X1, X2, X3 = X_train[:100], X_train[100:200], X_train[200:]
y1, y2, y3 = y_train[:100], y_train[100:200], y_train[200:]

clients = [(X1, y1), (X2, y2), (X3, y3)]

# Federated Learning
round_accuracies = []
global_model = get_model()

for r in range(5):
    client_weights = []
    client_sizes = []

    for X, y in clients:
        coef, intercept = train_client(X, y)
        client_weights.append((coef, intercept))
        client_sizes.append(len(y))

    global_model = aggregate(client_weights, client_sizes)

    acc = global_model.score(X_test, y_test)
    round_accuracies.append(acc)

    print(f"Round {r+1} Accuracy: {acc}")

# Plot graph
plt.plot(range(1, 6), round_accuracies)
plt.xlabel("Rounds")
plt.ylabel("Accuracy")
plt.title("Federated Learning Performance")
plt.show()

# Evaluation
y_pred = global_model.predict(X_test)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))
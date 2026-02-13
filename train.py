from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
import joblib

data = load_iris()
X, y = data.data, data.target

model = LogisticRegression(max_iter=200)
model.fit(X, y)

joblib.dump(model, "model.pkl")

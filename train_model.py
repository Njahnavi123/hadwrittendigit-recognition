import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import pickle

# Load dataset
data = pd.read_csv("dataset/digits.csv")

X = data.drop("label", axis=1)
y = data["label"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = GaussianNB()
model.fit(X_train, y_train)

# Save model
pickle.dump(model, open("model/digit_model.pkl", "wb"))

print("Digit recognition model trained successfully")

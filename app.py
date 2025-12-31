import pickle
import numpy as np

# Load trained model
model = pickle.load(open("model/digit_model.pkl", "rb"))

def predict_digit(pixels):
    pixels = np.array(pixels).reshape(1, -1)
    prediction = model.predict(pixels)
    return prediction[0]

if __name__ == "__main__":
    print("Enter 4 pixel values separated by space:")
    pixels = list(map(int, input().split()))
    print("Predicted Digit:", predict_digit(pixels))

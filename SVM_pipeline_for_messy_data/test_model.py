import pickle
import numpy as np

# load model
model = pickle.load(open("fraud_model.pkl", "rb"))

# example input (random values)
sample = np.random.rand(1, 30)

prediction = model.predict(sample)

print("Prediction:", prediction)
from keras.models import load_model
import numpy as np

# Load the saved model
loaded_model = load_model('voltage_classification_model2.h5')

# Define a function to predict the condition based on voltage input
def predict_condition(voltage):
    # Reshape voltage data
    voltage = np.array(voltage).reshape(-1, 1, 1)

    # Predict the condition
    prediction = loaded_model.predict(voltage)

    # Decode the prediction
    if prediction[0][0] > prediction[0][1]:
        return 'short-circuit'
    else:
        return 'normal'

# Accept user input for voltage
while True:
    try:
        voltage_input = float(input("Enter the voltage value: "))
        predicted_condition = predict_condition(voltage_input)
        print("Predicted condition:", predicted_condition)
    except ValueError:
        print("Please enter a valid numerical value for voltage.")

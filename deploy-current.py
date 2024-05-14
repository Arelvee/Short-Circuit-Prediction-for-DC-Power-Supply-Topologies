from keras.models import load_model
import numpy as np

# Load the saved model
loaded_model = load_model('linearcurrent_classification_model.h5')

# Define a function to predict the condition based on current input
def predict_condition(current):
    # Reshape current data
    current = np.array(current).reshape(-1, 1, 1)

    # Predict the condition
    prediction = loaded_model.predict(current)


    if prediction[0][0] > prediction[0][1]:
        return 'short-circuit'
    else:
        return 'normal'

# Accept user input for current
while True:
    try:
        current_input = float(input("Enter the current value: "))
        predicted_condition = predict_condition(current_input)
        print("Predicted condition:", predicted_condition)
    except ValueError:
        print("Please enter a valid numerical value for current.")

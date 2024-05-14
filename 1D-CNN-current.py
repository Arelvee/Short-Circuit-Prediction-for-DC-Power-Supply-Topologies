import pandas as pd
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, average_precision_score
from keras.callbacks import EarlyStopping
import numpy as np

# Load dataset
df = pd.read_csv('pywavelet_linearCurrent7.csv', header=None, skiprows=1)

# Assuming column 0 is voltage and column 1 is labels
current = df.iloc[:, 0].values.astype(float)
labels = df.iloc[:, 1].map({'short-circuit': 0, 'normal': 1}).values

# Reshape data for Conv1D
current = current.reshape((-1, 1, 1))

# One-hot encode labels
labels = to_categorical(labels)

# Split dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(current, labels, test_size=0.2, random_state=42)

# Define a Sequential model
model = Sequential()

# Add the first Convolutional layer
model.add(Conv1D(filters=32, kernel_size=1, activation='relu', input_shape=(1, 1)))

# Add a MaxPooling layer
model.add(MaxPooling1D(pool_size=1))

# Flatten the output to feed into a Dense layer
model.add(Flatten())

# Add a Dense layer
model.add(Dense(32, activation='relu'))

# Add Dropout layer
model.add(Dropout(0.2))

# Add the output layer
model.add(Dense(2, activation='softmax'))

# Compile the model with a lower learning rate
optimizer = Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Implementing early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=56, restore_best_weights=True)

# Train the model with early stopping and introduce Gaussian noise to the input data
# Define a function to add Gaussian noise to the input data
# def add_gaussian_noise(data, mean=1, std=0.01):
#     noise = np.random.normal(mean, std, data.shape)
#     return data + noise

# Train the model with noisy input data
history = model.fit((X_train), y_train, batch_size=32, epochs=56, validation_split=0.2, callbacks=[early_stopping], verbose=1)

# Plotting training history
plt.figure(figsize=(12, 6))

# Plot accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# Evaluate the model on test data without noise
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

# Calculate metrics
accuracy = accuracy_score(y_true_classes, y_pred_classes)
ap = average_precision_score(y_test, y_pred, average='macro')


# Define the custom formatting function
def format_metric(value):
    if value == 1.0:
        return "100%"
    else:
        return "{:.2f}".format(value * 100) + "%"

# Subtract the desired values from the metric values before formatting
accuracy_formatted = accuracy
ap_formatted = ap

# Print metrics with custom formatting
print("Accuracy:", format_metric(accuracy_formatted))
print("mAP:", format_metric(ap_formatted))
# Plot confusion matrix
conf_matrix = confusion_matrix(y_true_classes, y_pred_classes)
plt.figure(figsize=(8, 6))
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(len(['short-circuit', 'normal']))
plt.xticks(tick_marks, ['short-circuit', 'normal'], rotation=45)
plt.yticks(tick_marks, ['short-circuit', 'normal'])

thresh = conf_matrix.max() / 2.
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        plt.text(j, i, format(conf_matrix[i, j], 'd'),
                ha="center", va="center",
                color="white" if conf_matrix[i, j] > thresh else "black")

plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

# Plotting test data
plt.figure(figsize=(8, 6))
plt.scatter(X_test.squeeze(), y_test[:, 1], label='Actual', color='blue', alpha=0.5)
plt.scatter(X_test.squeeze(), y_pred[:, 1], label='Predicted', color='red', alpha=0.5)
plt.title('Test Data: Actual vs Predicted')
plt.xlabel('Voltage')
plt.ylabel('Probability of being normal')
plt.legend()
plt.show()
# Display model summary
model.summary()

# # Save the model
model.save('linearcurrent_classification_model.h5')
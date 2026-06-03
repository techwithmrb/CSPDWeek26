import tensorflow as tf
import numpy as np

print(f"TensorFlow version: {tf.__version__}")

# Define a simple sequential model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=[1])
])

# Compile the model
model.compile(optimizer='sgd', loss='mean_squared_error')

# Display the model summary
model.summary()


def trainModel(x_list, y_list, iterations,verbose=0):
    x = np.array(x_list, dtype=float)
    y = np.array(y_list, dtype=float)
    history = model.fit(x, y, epochs=iterations)
    print("\nTraining complete!")

def predictModel(x_input):
    new_x = np.array([x_input], dtype=float)
    predicted_y = model.predict(new_x)
    outputStr = f"For x = {new_x[0]}, the model predicts y = {predicted_y[0][0]:.4f}"
    return outputStr

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# List of file paths
file_paths = ['/Users/mingyuan/Downloads/fd01-sample1.csv', '/Users/mingyuan/Downloads/fd01-sample2.csv',
              '/Users/mingyuan/Downloads/fd02-sample1.csv','/Users/mingyuan/Downloads/fd02-sample2.csv',
              '/Users/mingyuan/Downloads/fd03-sample1.csv','/Users/mingyuan/Downloads/fd03-sample2.csv',
              '/Users/mingyuan/Downloads/fd04-sample1.csv','/Users/mingyuan/Downloads/fd04-sample2.csv',
              '/Users/mingyuan/Downloads/fd05-sample1.csv','/Users/mingyuan/Downloads/fd05-sample2.csv',
              ]

data = pd.concat([pd.read_csv(f) for f in file_paths])

data.drop('Amount', axis=1, inplace=True)

data = data.sample(frac=1).reset_index(drop=True)

X = data.drop('Class', axis=1)
y = data['Class']

X = X.values
y = y.values

split_index = int(0.8 * len(X))
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]



model = Sequential([
    Dense(16, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(16, activation='relu'),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=20, batch_size=32)

test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc}")

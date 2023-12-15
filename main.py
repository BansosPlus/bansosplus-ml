import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow.keras.optimizers import RMSprop

df = pd.read_excel('data/dummy_bansos.xlsx')

X = df.drop('status', axis=1)
y = df['status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)
X_train, X_cv, y_train, y_cv = train_test_split(X,y,test_size = 0.25,train_size =0.75, shuffle=True)

print("Training set shape:", X_train.shape, y_train.shape)
print("Validation set shape:", X_cv.shape, y_cv.shape)
print("Testing set shape:", X_test.shape, y_test.shape)

model = Sequential([
    Dense(8, activation='relu', input_shape=(9,)),
    Dense(16, activation='relu'),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])


model.compile(optimizer=RMSprop(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
# model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_cv, y_cv))

print(model.evaluate(X_test, y_test))

model.save("model_bansos.h5")

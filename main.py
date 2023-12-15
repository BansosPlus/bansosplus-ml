import pathlib

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow.keras.optimizers import RMSprop

def build_model():
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

    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_cv, y_cv))

    print(model.evaluate(X_test, y_test))

    return model

def convert_to_tflite(model):
    export_dir = 'saved_model/1'
    tf.saved_model.save(model,export_dir=export_dir)

    converter = tf.lite.TFLiteConverter.from_saved_model(export_dir)
    converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
    tflite_model = converter.convert()

    tflite_model_file = pathlib.Path('./model.tflite')
    tflite_model_file.write_bytes(tflite_model)

    return tflite_model

def test_model(tflite_model):
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    input_index = interpreter.get_input_details()[0]["index"]
    output = interpreter.tensor(interpreter.get_output_details()[0]['index'])

    input_data_sample = pd.DataFrame({
        'penghasilan': [1],
        'jumlah_makan': [2],
        'berobat': [0],
        'tanggungan': [2],
        'bahan_bakar': [1],
        'jumlah_aset': [1],
        'luas_lantai': [1],
        'jenis_dinding': [1],
        'pendidikan': [1],
    }, index=[0])

    input_data_sample = np.array(input_data_sample).astype(np.float32)

    interpreter.set_tensor(input_index, input_data_sample)

    interpreter.invoke()

    tflite_output = output()[0][0]

    print("TFLite Model Prediction:", tflite_output)

if __name__ == '__main__':
    model = build_model()
    tflite_model = convert_to_tflite(model)
    test_model(tflite_model)

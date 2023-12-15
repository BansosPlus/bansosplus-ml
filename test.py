import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_model(model_path):
    model = tf.keras.models.load_model(model_path)
    return model

def predict_bansos(model, input_data):
    predictions = model.predict(input_data)
    return predictions

model_path = "model_bansos.h5"
loaded_model = load_and_preprocess_model(model_path)

new_data = pd.DataFrame({
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

predictions = predict_bansos(loaded_model, new_data)

print(predictions)

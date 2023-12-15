import requests
import tensorflow as tf
import pandas as pd
import numpy as np

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title='API for bansosplus credit scoring')
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"])

def return_format(code, message, data, is_success=True):
    if type(data) == int:
        ret_data = data
    else:
        ret_data = data if len(data) > 0 else {}
    
    return {
        'code': code,
        'success': is_success,
        'message': message,
        'data': ret_data
    }

def load_and_preprocess_model(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

def predict_bansos(interpreter, input_data,):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_data = np.array(input_data, dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    predictions = interpreter.get_tensor(output_details[0]['index'])
    return predictions

@app.get("/")
def root():    
    return return_format(200, 'Successfully doing test', 'Hello World')

@app.post("/api/predict_new_user")
def predict_new_user(bansos_registration_id : int, penghasilan : int, jumlah_makan : int, berobat : int, tanggungan : int, bahan_bakar : int, jumlah_aset : int, luas_lantai : int, jenis_dinding : int, pendidikan: int):
    
    new_data = pd.DataFrame({
        'penghasilan': [penghasilan],
        'jumlah_makan': [jumlah_makan],
        'berobat': [berobat],
        'tanggungan': [tanggungan],
        'bahan_bakar': [bahan_bakar],
        'jumlah_aset': [jumlah_aset],
        'luas_lantai': [luas_lantai],
        'jenis_dinding': [jenis_dinding],
        'pendidikan': [pendidikan],
    }, index=[0])

    tflite_model_path = "model.tflite"  # Replace with the path to your .tflite file
    loaded_tflite_model = load_and_preprocess_model(tflite_model_path)

    predictions = predict_bansos(loaded_tflite_model, new_data)
    print(predictions)
    score = float(predictions[0])
    print(score)
    decisions = {}
    
    if score >= 0.75:
        decisions['id'] = bansos_registration_id
        decisions['score'] = round(score, 4)
        decisions['status'] = "ACCEPTED"
        x = requests.put(f'http://35.202.238.22:8001/api/bansos-registration/accept?bansos_registration_id={bansos_registration_id}')

    else:
        decisions['id'] = bansos_registration_id
        decisions['score'] = round(score, 4)
        decisions['status'] = "REJECTED"
        x = requests.put(f'http://35.202.238.22:8001/api/bansos-registration/reject?bansos_registration_id={bansos_registration_id}')  

    return return_format(200, 'Successfully doing prediction for new user', decisions)

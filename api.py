import requests

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

@app.get("/api/credit_scoring")
def main():
    return return_format(200, 'Successfully doing test', 'Hello World')

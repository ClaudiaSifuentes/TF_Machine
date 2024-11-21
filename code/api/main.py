import os
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from source.process import process, to_number
from tensorflow.keras.models import load_model
from io import StringIO
from sklearn.preprocessing import StandardScaler

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MAIN_DIR = os.path.dirname(os.getcwd())
MODEL_DIR = os.path.join(MAIN_DIR, 'models')
DATA_DIR = os.path.join(MAIN_DIR, 'data')

model = load_model(os.path.join(MODEL_DIR, 'model.keras'))
scaler = StandardScaler()
scaler.fit(process(pd.read_csv(os.path.join(DATA_DIR, 'merged_data.csv')), scale=False))

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/data/")
async def upload_file(file: UploadFile = File(...)):
    contents = await file.read()
    df = pd.read_csv(StringIO(contents.decode('utf-8')))
    df = process(df, scaler=scaler)
    predictions = (model.predict(df) > 0.5).astype(int)
    print(predictions)
    df['predictions'] = predictions
    result = df['predictions'].to_dict()
    return JSONResponse(content=result)

@app.post("/single/")
async def single(data: dict):
    print(data)
    df = pd.DataFrame(data, index=[0])
    df = to_number(df)
    df = process(df, scaler=scaler)
    result = {0 : int((model.predict(df) > 0.5)[0][0])}
    print(result)
    return JSONResponse(content=result)
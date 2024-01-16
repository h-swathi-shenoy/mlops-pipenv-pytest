from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.model import predict
from src.preprocessor import LogScaling, TransformationPipeline
import json
import uvicorn

app = FastAPI()


@app.get("/get")
async def get():
    return {"status": "OK"}


@app.post("/predict/", status_code=200)
def get_prediction(payload: dict):
    test_arr = json.loads(payload["test_data"])
    prediction = predict(test_arr)

    if not prediction:
        raise HTTPException(status_code=400, detail="Model not found.")

    response_object = prediction
    return response_object


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8080)

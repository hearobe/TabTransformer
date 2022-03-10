from fastapi import FastAPI
from Utils import *

app = FastAPI()

@app.get("/")
async def root():
    return {"/accuracy":"Retrieve accuracy and other prediction measurements",
            "/accuracy/<a_number>":"Run the model with your desired number of transformers"}


@app.get("/accuracy")
async def predict():
    prepare_tab, X_train, X_test, y_train, y_test = prepare_data()
    model = set_model(prepare_tab, 0)
    result = run_experiment_and_save(
        model,
        X_train,
        X_test,
        y_train,
        y_test
    )
    print(result)
    return result

@app.get("/accuracy/{depth}")
async def predict(depth: int):
    prepare_tab, X_train, X_test, y_train, y_test = prepare_data()
    model = set_model(prepare_tab, depth)
    result = run_experiment_and_save(
        model,
        X_train,
        X_test,
        y_train,
        y_test
    )
    print(result)
    return result

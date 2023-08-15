import json
import numpy
from azureml.core.model import Model
import joblib
import pickle

import warnings

def init():

    global model
    model_path = Model.get_model_path(model_name="sklearn_regression_model.pkl")
    model = joblib.load(model_path)
    

def run(raw_data, request_readers):
    
    data = json.loads(raw_data)["data"] # transform json to python list
    data = numpy.array(data)
    result = model.predict(data)

    return {'result': result.tolist()}


init()


test_row = '{"data":[[1,2,3,4,5,6,7,8,9,10],[10,9,8,7,6,5,4,3,2,1]]}'
request_header = {}
prediction = run(test_row, request_header)
print("Test result: ", prediction)


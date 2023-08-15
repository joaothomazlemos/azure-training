# %% [markdown]
# # Score Data with a Ridge Regression Model Trained on the Diabetes Dataset

# %% [markdown]
# This notebook loads the model trained in the Diabetes Ridge Regression Training notebook, prepares the data, and scores the data.

# %%
import json
import numpy
from azureml.core.model import Model
import joblib
import pickle

import warnings
warnings.filterwarnings("ignore")

# %% [markdown]
# ## Load Model

# %%
model_path = Model.get_model_path(model_name="sklearn_regression_model.pkl")
model = joblib.load(model_path)

# %% [markdown]
# ## Prepare Data

# %%
raw_data = '{"data":[[1,2,3,4,5,6,7,8,9,10],[10,9,8,7,6,5,4,3,2,1]]}' # 2  samples input data ( even though has nothing in common with the training data)

data = json.loads(raw_data)["data"] # transform json to python list
data = numpy.array(data)

# %% [markdown]
# ## Score Data

# %%
request_headers = {}

result = model.predict(data)
print("Test result: ", {"result": result.tolist()})

# %%
data[0:1]

# %%
model.predict(data[0:1])



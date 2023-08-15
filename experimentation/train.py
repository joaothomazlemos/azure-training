from sklearn.datasets import load_diabetes
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import joblib
import pandas as pd


def split_data(df):
    """Split data into train and test sets"""
    X = df.drop('Y', axis=1).values
    y = df['Y'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    data = {'train': {'X': X_train, 'y': y_train},
            'test': {'X': X_test, 'y': y_test}}
    return data



def train_model(data, args):
    'which takes the parameters data and args and returns a trained model.'
    reg_model = Ridge(**args)
    reg_model.fit(data['train']['X'], data['train']['y'])

    return reg_model



def get_model_metrics(reg_model, data):
    preds = reg_model.predict(data['test']['X'])
    mse = mean_squared_error(preds, data['test']['y'])
    metrics = {'mse': mse}
    return metrics



def main():
    #load data
    sample_data = load_diabetes()
    df = pd.DataFrame(
        data=sample_data.data,
        columns=sample_data.feature_names)
    df['Y'] = sample_data.target

    # Split Data into Training and Validation Sets
    data = split_data(df)

    # Train Model on Training Set
    args = {
        "alpha": 0.5
    }
    reg = train_model(data, args)

    # Validate Model on Validation Set
    metrics = get_model_metrics(reg, data)

    # Save Model
    model_name = "sklearn_regression_model.pkl"

    joblib.dump(value=reg, filename=model_name)



if __name__ == '__main__':
    main()


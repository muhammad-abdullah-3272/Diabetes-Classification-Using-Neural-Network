#------------------------------------------------------------------------------
# Importing Necessary libraires  
#------------------------------------------------------------------------------
import pandas as pd

#------------------------------------------------------------------------------
# Importing Model from the file 'Model'
#------------------------------------------------------------------------------
from Model import Model

# Instantiate the Model
model = Model()

# Prediction Function
def prediction(X):
    '''
    Parameters
    ----------
    X : TYPE Numpy Array
        DESCRIPTION. Data Features

    Returns
    -------
    TYPE Numpy Array / Int
        DESCRIPTION. Predicted Output
    '''
    # Returns Prediction    
    return model.Predict(X)


if __name__ == "__main__":

    # Unseen Data Features
    features = pd.read_csv('New_Examples.csv', encoding='unicode_escape')
    

    # Predict the output of the given Features 
    y = prediction(features)
    print("\nPredictions:\n", y)

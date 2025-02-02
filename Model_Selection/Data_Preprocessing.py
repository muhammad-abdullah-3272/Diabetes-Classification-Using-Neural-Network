## Importing Necessary Libraries
import numpy as np
import pandas as pd
np.random.seed(9)

pd.options.mode.chained_assignment = None 

## Reading the Dataset 
data = pd.read_csv('data02.csv', encoding='unicode_escape')

def dataReplace(data):
    '''
    This fucntion converts string data type to int data type.
    The model takes the following arguments:
    
    Data (Features containing String data type)
    
    returns:
    Data (Features converted to Int data type)
    '''
    
    # Converting Date and Time to Datetime object
    data['Date'] = pd.to_datetime(data['Date'], format = '%m/%d/%Y')
    data['Time'] = pd.to_datetime(data['Time'], format = '%H:%M')
    
    # Converting the Datetime object to numeric data
    data['Date'] = pd.to_numeric(data['Date'])
    data['Time'] = pd.to_numeric(data['Time'])

    return data


def Scale(data):
    '''
    This fucntion performs data scaling from -1 to 1 using min-max critera.
    The model takes the following arguments:
    
    Data (numpy array): Input dataset
    
    returns:
    Scaled Data (numpy array)
    '''
    
    dataScale = 2*((data - data.min()) / (data.max() - data.min())) - 1    # Feature Scaling from -1 to 1
    dataScale['Outcome'] = data['Outcome']                                 # Not applying Scaling on Y
    
    for i in range(len(dataScale)):
        if dataScale['Outcome'][i] <= 120:
            dataScale['Outcome'][i] = 0
        else:
            dataScale['Outcome'][i] = 1
    return dataScale

data = dataReplace(data)                          # Calling the Data Conversion Function and assigning it to variable data
data = Scale(data)                                # Calling the Feature Scaling Function and assigning it to variable data

# Splitting the Dataset into Train set(60%), Cross Validation set(20%) and Test set(20%)
train, val, test = np.split(data.sample(frac=1), [int(0.7 * len(data)), int(0.85 * len(data))])
print("Dataset: ", data.shape)
print("Training Set: ", train.shape)
print("Validation Set: ",val.shape)
print("Test Set: ",test.shape)



X_data = ["Date", "Time", "Code"]                                 # Extracting Features
Y_data = ["Outcome"]                                              # Extracting Labels

X_train = train[X_data]                                           # Assigning Features to X_train               
Y_train = train[Y_data]                                           # Assigning Features to Y_train

X_val = val[X_data]                                               # Assigning Features to X_val
Y_val = val[Y_data]                                               # Assigning Features to Y_val

X_test = test[X_data]                                             # Assigning Features to X_test
Y_test = test[Y_data]                                             # Assigning Features to Y_test


X_train = X_train.values                                          # Extracting values from X_train
Y_train = Y_train.values                                          # Extracting values from Y_train

X_val = X_val.values                                              # Extracting values from X_val
Y_val = Y_val.values                                              # Extracting values from Y_val

X_test = X_test.values                                            # Extracting values from X_test
Y_test = Y_test.values                                            # Extracting values from Y_test


print("Shape of X_train : ", X_train.shape)
print("Shape of Y_train : ", Y_train.shape)

print("Shape of X_val : ", X_val.shape)
print("Shape of Y_val : ", Y_val.shape)

print("Shape of X_test : ", X_test.shape)
print("Shape of Y_test : ", Y_test.shape)


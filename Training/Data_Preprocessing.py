#------------------------------------------------------------------------------
# Importing Necessary libraires  
#------------------------------------------------------------------------------
import numpy as np
import pandas as pd
np.random.seed(5)

pd.options.mode.chained_assignment = None 


#==============================================================================
# Class Data
#==============================================================================
class Data:
    #--------------------------------------------------------------------------
    # Replace the String Values in the data to Int data type
    #--------------------------------------------------------------------------
    def Replace(self, data):
        '''
        Parameters
        ----------
        data : TYPE pandas.core.frame.DataFrame
            DESCRIPTION. Data Frame (Features containing String data type)

        Returns
        -------
        None.
        '''

        self.data = data
        
        # Converting Date and Time to Datetime object
        self.data['Date'] = pd.to_datetime(self.data['Date'], format = '%m/%d/%Y')
        self.data['Time'] = pd.to_datetime(self.data['Time'], format = '%H:%M')
        
        # Converting the Datetime object to numeric data
        self.data['Date'] = pd.to_numeric(self.data['Date'])
        self.data['Time'] = pd.to_numeric(self.data['Time'])      
    
    #--------------------------------------------------------------------------
    # Scale the data. Normalize it from -1 to 1
    #--------------------------------------------------------------------------
    def Scale(self, data_scale=None):
        '''
        Parameters
        ----------
        data_scale : TYPE Numpy Array, optional
            DESCRIPTION. Dataset provided by the user. The default is None.

        Returns
        -------
        None.
        '''

        # If data is provided by the user
        if data_scale is not None:
            # Feature Scaling from -1 to 1
            self.Scaled_Data = 2*((data_scale - data_scale.min()) / (data_scale.max() - data_scale.min())) - 1
            # Not applying Scaling on Y
            self.Scaled_Data['Outcome'] = data_scale['Outcome']  
            
            for i in range(len(self.Scaled_Data)):
                if self.Scaled_Data['Outcome'] <= 120:
                    self.Scaled_Data['Outcome'][i] = 0
                else:
                    self.Scaled_Data['Outcome'][i] = 1                 
        
        # Else, scale the above data
        else:    
            # Feature Scaling from -1 to 1
            self.Scaled_Data = 2*((self.data - self.data.min()) / (self.data.max() - self.data.min())) - 1 
            # Not applying Scaling on Y
            self.Scaled_Data['Outcome'] = self.data['Outcome']    

            for i in range(len(self.Scaled_Data)):
                if self.Scaled_Data['Outcome'][i] <= 120:
                    self.Scaled_Data['Outcome'][i] = 0
                else:
                    self.Scaled_Data['Outcome'][i] = 1  
        
        
    #--------------------------------------------------------------------------
    # Split the Dataset in Train, Validation and Test sets
    #--------------------------------------------------------------------------
    def Split(self, Train, Val_Test, split_data=None):
        '''
        Parameters
        ----------
        Train : TYPE Int
            DESCRIPTION. %age for Training data set distribution
        Val_Test : TYPE Int
            DESCRIPTION. %age for validation and test data sets distribution
        split_data : TYPE Numpy Array, optional
            DESCRIPTION. Dataset provided by the user. The default is None.

        Returns
        -------
        X_train, Y_train, X_val, Y_val, X_test, Y_test : TYPE Numpy Array
            DESCRIPTION. Training, Validation and Test sets' Features and Labels
        '''
        if split_data is not None:
            # Splitting the Dataset into Train set(70%), Cross Validation set(15%) and Test set(15%).
            train, val, test = np.split(split_data.sample(frac=1), [int(Train * len(split_data)), int(Val_Test * len(split_data))])
        else:    
            # Splitting the Dataset into Train set(70%), Cross Validation set(15%) and Test set(15%).
            train, val, test = np.split(self.Scaled_Data.sample(frac=1), [int(Train * len(self.Scaled_Data)), int(Val_Test * len(self.Scaled_Data))])
    

        X_data = ["Date", "Time", "Code"]                             # Extracting Features
        Y_data = ["Outcome"]                                          # Extracting Labels
        
        X_train = train[X_data]                                       # Assigning Features to X_train               
        Y_train = train[Y_data]                                       # Assigning Features to Y_train
        
        X_val = val[X_data]                                           # Assigning Features to X_val
        Y_val = val[Y_data]                                           # Assigning Features to Y_val
        
        X_test = test[X_data]                                         # Assigning Features to X_test
        Y_test = test[Y_data]                                         # Assigning Features to Y_test
        
        X_train = X_train.values                                      # Extracting values from X_train
        Y_train = Y_train.values                                      # Extracting values from Y_train
        
        X_val = X_val.values                                          # Extracting values from X_val
        Y_val = Y_val.values                                          # Extracting values from Y_val
        
        X_test = X_test.values                                        # Extracting values from X_test
        Y_test = Y_test.values                                        # Extracting values from Y_test
        
        print("Shape of X_train : ", X_train.shape)
        print("Shape of Y_train : ", Y_train.shape)
        
        print("Shape of X_val : ", X_val.shape)
        print("Shape of Y_val : ", Y_val.shape)
        
        print("Shape of X_test : ", X_test.shape)
        print("Shape of Y_test : ", Y_test.shape)
        
        return X_train, Y_train, X_val, Y_val, X_test, Y_test

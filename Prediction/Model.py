#------------------------------------------------------------------------------
# Importing Necessary libraires  
#------------------------------------------------------------------------------
import numpy as np
import pandas as pd
from datetime import datetime as dt
import pickle

#------------------------------------------------------------------------------
# Importing Classes from the file 'Classes'
#------------------------------------------------------------------------------
from Classes import Dense_Layer, Input_Layer, ReLU_Activation, Sigmoid_Activation

#==============================================================================
# Class Model
#==============================================================================
class Model():
    
    def __init__(self):
        # Create a List of Network Objects
        self.layers = []
        
    #--------------------------------------------------------------------------
    # Add Layers to the Model
    #--------------------------------------------------------------------------
    def Add(self, layer):
        '''
        Parameters
        ----------
        layer : DESCRIPTION. Layers can be added in the network due to this function

        Returns
        -------
        None.
        '''
        self.layers.append(layer)
        
    #--------------------------------------------------------------------------
    # Finalize the Model
    #--------------------------------------------------------------------------
    def Finalize(self):
        # Create and ser the Input Layer
        self.input_layer = Input_Layer()
        
        # # Count all the layers in the network
        layer_count = len(self.layers)
        
        # Initialize a list containing all trainable layers
        self.trainable_layers = []
        
        # Iterate the layers
        for i in range(layer_count):
            
            # If it is the first layer, then the previous layer object is the input layer
            if i == 0:
                self.layers[i].prev = self.input_layer
                self.layers[i].next = self.layers[i+1]
                
            # Consider all layers except for the first and the last layer 
            elif i < layer_count - 1:
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.layers[i+1]
                
            # After last layer, the next object is the loss. Also save this object which is the output of the model
            else:
                self.layers[i].prev = self.layers[i-1]
                self.output_layer_activation = self.layers[i]
                
            # If layer contains an attribute 'weights', then its a trainable layer, add it to the list of trainabe layers
            if hasattr(self.layers[i], 'weights'):
                self.trainable_layers.append(self.layers[i])

    #--------------------------------------------------------------------------
    # Perform Forward Propagation
    #--------------------------------------------------------------------------        
    def Forward(self, X, training):
        '''
        Parameters
        ----------
        X : TYPE Numpy Array
            DESCRIPTION. Data set features
        training : TYPE Bool
            DESCRIPTION. Checks wether which dataset to train on and on which to perform forward pass only

        Returns
        -------
        layer.output: TYPE Numpy Array List
            DESCRIPTION. Performs forward propagation layer wise and returns list
        '''
        # Call Forward propagation, this will set the O/P property that the first layer in previos object is expecting
        self.input_layer.Forward(X, training)
        
        # Call Forward method in chain. Pass O/P of the previous layer as a parameter 
        for layer in self.layers:
            layer.Forward(layer.prev.output, training)
        return layer.output
    
    
    #--------------------------------------------------------------------------
    # Predict an unseen data on the Model with optimized parameters
    #--------------------------------------------------------------------------
    def Predict(self, X=None):
        '''
        Parameters
        ----------
        X : TYPE Numpy Array, optional
            DESCRIPTION. Unseen Data to predict their outputs. The default is None.

        Returns
        -------
        predictions : TYPE numpy array or scalar value if single sample is provided
            DESCRIPTION. Predicted value on the bases of training.
        '''
        # IF Data is provided
        if X is not None:
            
            self.X = X
            print(X.head())
            # Add Layers on which the Model has Trained
            self.Add(Dense_Layer(3, 7))
            self.Add(ReLU_Activation())
            self.Add(Dense_Layer(7, 1))
            self.Add(Sigmoid_Activation())
            
            # Finalize the Model
            self.Finalize()
            
            # Load the Optimized Parameters
            self.Load_Params('Diabetes.params')
            
            # Converting Date and Time to Datetime object
            self.X['Date'] = pd.to_datetime(self.X['Date'], format = '%m/%d/%Y')
            self.X['Time'] = pd.to_datetime(self.X['Time'], format = '%H:%M')
            
            # Converting the Datetime object to numeric data
            self.X['Date'] = pd.to_numeric(self.X['Date'])
            self.X['Time'] = pd.to_numeric(self.X['Time'])
            
            # Load Min Max of the Dataset
            Max = np.load('Max.npy')
            Min = np.load('Min.npy')

            # Feature Scaling from -1 to 1
            self.scaled_data = 2*((self.X - Min) / (Max - Min)) - 1

            output = []
            
            # Perform forward prop without training
            sample_output = self.Forward(self.scaled_data, training=False)
            output.append(sample_output)
            
            # Get Predictions
            predictions = self.output_layer_activation.Predictions(np.vstack(output))
            
            return predictions
    
    #--------------------------------------------------------------------------
    # Update the Model with New Parameters
    #--------------------------------------------------------------------------
    def Set_Params(self, params):
        '''
        Parameters
        ----------
        params : TYPE List Numpy Array
            DESCRIPTION. Weights and Biases are set for predictions
                         Parameters are set layer wise

        Returns
        -------
        None.
        '''
        # Iterate over the Params and Layers and update each Layer with each Set of Params
        for param_set, layer in zip(params, self.trainable_layers):
            layer.Set_Params(*param_set)
            
     
    #--------------------------------------------------------------------------
    # Load the Weights and Update the Model instance with Them
    #--------------------------------------------------------------------------
    def Load_Params(self, path):
        '''
        Parameters
        ----------
        path : TYPE file. Binary Read Mode.
            DESCRIPTION. Parametrs are loaded from this file.

        Returns
        -------
        None.
        '''
        with open (path, 'rb') as f:
            self.Set_Params(pickle.load(f))
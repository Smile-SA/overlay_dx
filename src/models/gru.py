"""Gated Recurrent Unit, is a simplified version of LSTM that is also used for time series forecasting.
"""
# import libraries
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, GRU
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import subprocess
import numpy as np
import pandas as pd

# Specify the version you want to install
version = '==3.14.0'
# Run the pip install command using subprocess
subprocess.check_call(['pip', 'install', 'protobuf' + version])

class gru():
    """This class is used to forecast time series data using Gated Recurrent Unit neural network. \n
    
    Attributes: \n
    df (pandas dataframe): pandas dataframe with the time series as columns \n
    target (str): name of target column to forecast \n
    test_size (float): the percentage size of the test set from 0 to 1  \n
    plot_learning_curves_bool (bool): boolean to plot learning curves or not, default is False  \n

    Import and usage : \n
    from src.models.gru import gru \n
    gru_model = gru(df, target, test_size, plot_learning_curves_bool) \n
    pred, test, train = gru_model.fit() \n
    forecast = gru_model.forecast(new_data) \n
    """ 

    def __init__(self, df, target, test_size, plot_learning_curves_bool = False, epochs = 200, batch_size = 30):
        """Initialise the class with the data, target variable, test size and plot learning curves boolean"""
        self.data = df
        self.target_variable = target
        self.test_size = test_size
        self.plot_learning_curves_bool = plot_learning_curves_bool
        self.trainX, self.testX, self.trainY, self.testY = self.train_test_sets()
        self.model = None
        self.epochs = epochs
        self.batch_size = batch_size
        self.create_model(None)
    

    def train_test_sets(self):
        #Split the data into train and test sets
        X = self.data.drop(self.target_variable,1)
        Y = self.data[self.target_variable]
        
        trainX, testX, trainY, testY = train_test_split(
        X,
        Y,
        test_size=self.test_size,
        random_state=42,
        shuffle=False
        )
        # Scale the features using MinMax scaler
        scaler = MinMaxScaler()
        trainX = scaler.fit_transform(trainX)
        testX = scaler.transform(testX)
        # Reshape the data to be 3D for GRU
        trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
        testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

        return trainX, testX, trainY, testY
 

    def create_model(self,custom_model):
        """Create the GRU model.
        By default, the model is a sequential model with 3 GRU layers and 3 dense layers.
        The loss function is mean squared error and the optimizer is adam.
        However, you can define your own architecture by providing your own Sequential model architecture from keras.
        """
        if custom_model is not None and isinstance(custom_model, Sequential):
            model = custom_model
            input_shape = (self.trainX.shape[1], self.trainX.shape[2])
            if model.layers[0].input_shape[1:] != input_shape:
                raise ValueError("The input shape of the custom model does not match self.trainX shape.",input_shape)
        else :
            # Define the GRU model
            model = Sequential()
            model.add(GRU(512, input_shape=(self.trainX.shape[1], self.trainX.shape[2]), return_sequences=True))
            # Add second GRU layer
            model.add(GRU(256, return_sequences=True))
            # Add third GRU layer
            model.add(GRU(128, return_sequences=False))
            #model.add(Dense(64, activation='linear'))
            #model.add(Dense(32, activation='linear'))
            model.add(Dense(8, activation='linear'))
            model.add(Dense(1, activation='linear'))
            model.compile(loss='mse', optimizer='adam')
        self.model = model

    def plot_learning_curves(self,history):
        #Plot the learning curves of the neural network
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.show()

    def plot_predictions(self, predictions):
        plt.plot(self.trainY, color="blue")
        plt.plot(self.testY, color="green")
        plt.plot(predictions, color="orange")
        plt.title('Predictions vs Actual Values')
        plt.legend(['Train', 'Test', 'Forecast'], loc='upper left')
        # set the colors: blue for training set, green for test set, orange for forecast
        plt.ylabel('values')
        plt.xlabel('time')
        plt.show()


    def fit(self):
        """This function is used to train, test and forecast time series data using gru neural networks. """

        # Fit the model to the training data
        history = self.model.fit(self.trainX, self.trainY, epochs=self.epochs, batch_size=self.batch_size, validation_data=(self.testX, self.testY), shuffle=False,callbacks=[
                EarlyStopping(monitor='val_loss', patience=5)])

        # Plot learning curves
        if self.plot_learning_curves_bool:
            self.plot_learning_curves(history)

        # Make predictions on the testing data
        predictions = self.model.predict(self.testX)

        # set the index of the predictions to the same as the testY
        predictions = pd.Series(predictions.flatten())
        predictions.index = self.testY.index

        # Plot the predictions vs the actual train and test values
        self.plot_predictions(predictions)

        return predictions, self.testY, self.trainY


    def forecast(self, new_data):
        """This function is used to forecast time series data using GRU neural networks on new_data. \n

        Parameters: \n
        new_data (numpy array): The new data to forecast on. The shape of the array should be (n_samples, n_timesteps, n_features). \n

        Returns: \n
        predictions (numpy array): The predictions of the model on the new data. The shape of the array is (n_samples, 1). \n
        """

        # Make sure that the number of features in the input data matches the number of features used to train the model and return an error if not.
        if new_data.shape[2] != self.trainX.shape[2]:
            raise ValueError(
                f"The number of features in the input data {new_data.shape[2]} does not match the number of features used to train the model {self.trainX.shape[2]}.")
        else:
            predictions = self.model.predict(new_data)
            print("predictions",predictions)
            return predictions


if __name__ == "__main__":
    
    import pandas as pd
    #Create a dataframe of with two time series columns
    df = pd.DataFrame({"Date": ['01-01-2000', '01-02-2000', '01-03-2000', '01-04-2000'] , "Close": [1,2,3,4], "Open": [2,3,4,5], "High": [3,4,5,6], "Low": [4,5,6,7]})
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date")
    print(df)
    #Initialise the class
    model = gru(df = df, target = "Close", test_size = 0.2, plot_learning_curves_bool = False)
    #Forecast the time series data
    test=model.fit()
    forecast = model.forecast(new_data =np.array([[[2,3,4]],[[3,4,5]]]))
    print(forecast)
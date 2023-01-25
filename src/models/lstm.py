"""Long Short Term Memory, LSTM is a type of recurrent neural network (RNN) that can learn long-term dependencies between time steps of sequence data.
"""
# import libraries
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import subprocess
import numpy as np
import pandas as pd
# Specify the version you want to install
version = '==3.14.0'
# Run the pip install command using subprocess
subprocess.check_call(['pip', 'install', 'protobuf' + version])

class lstm():
    """This class is used to forecast time series data using Long Short Term Memory. \n
     
    Attributes: \n
        df (pandas dataframe): pandas dataframe with the time series as columns \n
        target (str): name of target column to forecast \n
        test_size (float): the percentage size of the test set from 0 to 1 \n
        plot_learning_curves_bool (bool): boolean to plot learning curves or not, default is False \n
    
    Import and usage: \n
        from src.models.lstm import lstm \n
        lstm_model = lstm(df, target, test_size, plot_learning_curves_bool) \n
        predictions, test, train = lstm_model.fit() \n
        forecast = lstm_model.forecast(new_data) \n
    """ 

    def __init__(self, df, target, test_size, plot_learning_curves_bool = False, epochs = 200, batch_size = 30):
        """Initialise the class with the different attributes"""
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
        """This function is used to split the data into train and test sets."""
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
        
        trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
        testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
        return trainX, testX, trainY, testY

    def create_model(self,custom_model):
        """This method is used to create the LSTM model.
        By default, the model has 3 LSTM layers and 2 Dense layers.
        However, you can define your own architecture by calling this method and providing your own Sequential model architecture from keras.
        """
        if custom_model is not None and isinstance(custom_model, Sequential):
            model = custom_model
            input_shape = (self.trainX.shape[1], self.trainX.shape[2])
            if model.layers[0].input_shape[1:] != input_shape:
                raise ValueError("The input shape of the custom model does not match self.trainX shape.",input_shape)
        else :
            # Define the LSTM model
            model = Sequential()
            model.add(LSTM(512, input_shape=(self.trainX.shape[1], self.trainX.shape[2]), return_sequences=True))
            # Add second LSTM layer
            model.add(LSTM(256, return_sequences=True))
            # Add third LSTM layer
            model.add(LSTM(128, return_sequences=False))
            #model.add(Dense(64, activation='linear'))
            #model.add(Dense(32, activation='linear'))
            model.add(Dense(8, activation='linear'))
            model.add(Dense(1, activation='linear'))
            model.compile(loss='mse', optimizer='adam')
        self.model = model

    def plot_learning_curves(self,history):
        """This function is used to plot the learning curves of the neural network."""
        #Plot the learning curves of the neural network
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.show()

    def plot_predictions(self, predictions):
        """This function is used to plot the predictions vs the actual train and test values."""
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
        """This method is used to train, test and forecast time series data using LSTM neural networks. """

        # Fit the model to the training data
        history = self.model.fit(self.trainX, self.trainY, epochs=self.epochs, batch_size=self.batch_size , validation_data=(self.testX, self.testY), shuffle=False,callbacks=[
                EarlyStopping(monitor='val_loss', patience=10)])

        # Plot learning curves
        if self.plot_learning_curves_bool:
            self.plot_learning_curves(history)

        # Make predictions on the testing data
        predictions = self.model.predict(self.testX)
        # set the index of the predictions to the same as the testY
        predictions = pd.Series(predictions.flatten())
        predictions.index = self.testY.index
        print("predictions",predictions)

        # Plot the predictions vs the actual train and test values
        self.plot_predictions(predictions)

        return predictions, self.testY, self.trainY
    
    
    def forecast(self, new_data):
        """This method is used to forecast new time series data using the trained LSTM model \n
        Parameters: \n
        new_data: (Numpy array) The new input data to be forecasted. \n

        Returns: \n
        predictions:  (Numpy array) The forecasted values. \n"""

        # Reshape the input for forecasting
        #X_val = np.reshape(X_val.values, (X_val.shape[0], 1, X_val.shape[1]))

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
    df = pd.DataFrame({"Date": ['01-01-2000', '01-02-2000', '01-03-2000', '01-04-2000','01-05-2000','01-06-2000','01-07-2000','01-08-2000','01-09-2000'] , "Close": [1,2,9,4,5,6,7,15,9], "Open": [2,3,4,14,6,7,8,9,10], "High": [3,4,5,8,7,8,11,10,7], "Low": [4,5,6,7,8,9,10,14,13]})
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date")
    df = df.append(df.copy())
    print(df)
    print(df.columns.tolist())
    #Initialise the class
    model = lstm(df = df[["Close","Open","High","Low"]], target = "Close", test_size = 0.2, plot_learning_curves_bool = False)
    #Forecast the time series data
    test=model.fit()
    new_data = pd.DataFrame({"Date": ['01-10-2000', '01-11-2000', '01-12-2000'] , "Close": [np.nan,np.nan,np.nan], "Open": [11,12,13], "High": [12,13,14], "Low": [13,14,15]}).set_index("Date")
    forecast = model.forecast(new_data = new_data[["Open","High","Low"]].values.reshape(3,1,3))
    print(forecast)

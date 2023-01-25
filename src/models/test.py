# import libraries
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, GRU, Dense, Dropout
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import subprocess
import numpy as np
import pandas as pd

# Specify the version you want to install
version = '==3.14.0'
# Run the pip install command using subprocess
subprocess.check_call(['pip', 'install', 'protobuf' + version])

class lstm():
    """This class is used to forecast time series data using Long Short Term Memory or Gated Recurrent Unit.
     
    Attributes:
        df (pandas dataframe): pandas dataframe with the time series as columns
        target (str): name of target column to forecast
        test_size (float): the percentage size of the test set from 0 to 1
        plot_learning_curves_bool (bool): boolean to plot learning curves or not, default is False
        sequence_length (int): the length of input sequences

    """ 

    def __init__(self, df, target, test_size, plot_learning_curves_bool=False, sequence_length=10,
                 lstm_layers=3, lstm_units=512, gru_layers=0, gru_units=0,
                 dropout_rate=0.2, learning_rate=0.001):
        """Initialise the class with the different attributes"""
        self.data = df
        self.target_variable = target
        self.test_size = test_size
        self.plot_learning_curves_bool = plot_learning_curves_bool
        self.sequence_length = sequence_length
        self.lstm_layers = lstm_layers
        self.lstm_units = lstm_units
        self.gru_layers = gru_layers
        self.gru_units = gru_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.trainX, self.testX, self.trainY, self.testY = self.train_test_sets()
        self.model = None
        self.create_model(None)
    
    def train_test_sets(self):
        """This function is used to split the data into train and test sets."""
        #Split the data into train and test sets
        X = self.data.drop(self.target_variable, 1)
        Y = self.data[self.target_variable]

        # Scale the features using MinMax scaler
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)

        # Split the data into sequences
        sequences_X, sequences_Y = [], []
        for i in range(len(X) - self.sequence_length):
            sequences_X.append(X[i : i + self.sequence_length])
            sequences_Y.append(Y.iloc[i + self.sequence_length])

        sequences_X = np.array(sequences_X)
        sequences_Y = np.array(sequences_Y)

        trainX, testX, trainY, testY = train_test_split(
            sequences_X,
            sequences_Y,
            test_size=self.test_size,
            random_state=42,
            shuffle=False
        )

        return trainX, testX, trainY, testY

    def create_model(self, custom_model):
        """This method is used to create the LSTM or GRU model.
        By default, the model has 3 LSTM layers and 2 Dense layers.
        However, you can define your own architecture by calling this method and providing your own Sequential model architecture from keras.
        """
        if custom_model is not None and isinstance(custom_model, Sequential):
            model = custom_model
            input_shape = (self.trainX.shape[1], self.trainX.shape[2])
            if model.layers[0].input_shape[1:] != input_shape:
                raise ValueError("The input shape of the custom model does not match self.trainX shape.", input_shape)
        else:
            # Define the LSTM or GRU model
            model = Sequential()

            if self.lstm_layers > 0:
                model.add(LSTM(self.lstm_units, input_shape=(self.trainX.shape[1], self.trainX.shape[2]),
                               return_sequences=True))
                for _ in range(1, self.lstm_layers):
                    model.add(LSTM(self.lstm_units, return_sequences=True))

            if self.gru_layers > 0:
                model.add(GRU(self.gru_units, input_shape=(self.trainX.shape[1], self.trainX.shape[2]),
                              return_sequences=True))
                for _ in range(1, self.gru_layers):
                    model.add(GRU(self.gru_units, return_sequences=True))

            model.add(Dropout(self.dropout_rate))
            model.add(Dense(8, activation='linear'))
            model.add(Dense(1, activation='linear'))
            # Use the Adam optimizer and set the learning rate
            optimizer = Adam(learning_rate=self.learning_rate)
            model.compile(loss='mse', optimizer=optimizer)

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
        history = self.model.fit(self.trainX, self.trainY, epochs=200, batch_size=30, validation_data=(self.testX, self.testY), shuffle=False,callbacks=[
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
        """This method is used to forecast new time series data using the trained LSTM model"""
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


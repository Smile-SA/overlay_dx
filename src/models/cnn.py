"""Convolutional Neural Network, CNN is a type of neural network that can be used for time series forecasting by using convolutional layers.
"""
# import libraries
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten
from keras.callbacks import  EarlyStopping
import matplotlib.pyplot as plt
import subprocess
import numpy as np
import pandas as pd

# Specify the version you want to install
version = '==3.14.0'
# Run the pip install command using subprocess
subprocess.check_call(['pip', 'install', 'protobuf' + version])

class cnn():
    """This class is used to forecast time series data using Convolutional Neural Network. \n

    Attributes : \n
        df (pandas dataframe) : pandas dataframe with the time series as columns \n
        target (str) : name of target column to forecast \n
        test_size (float) : the percentage size of the test set from 0 to 1 \n
        plot_learning_curves_bool (bool) : boolean to plot learning curves or not, default is False \n

    Import and usage: \n
        from src.models.cnn import cnn \n
        cnn_model = cnn(df, target, test_size, plot_learning_curves_bool) \n
        pred, test, train = cnn_model.fit() \n
        forecast = cnn_model.forecast(new_data) \n

    """ 

    def __init__(self, df, target, test_size, plot_learning_curves_bool = False):
        #Initialise the class with the data, target variable, test size and plot learning curves boolean
        self.data = df
        self.target_variable = target
        self.test_size = test_size
        self.plot_learning_curves_bool = plot_learning_curves_bool
        self.trainX, self.testX, self.trainY, self.testY = self.train_test_sets()
        self.model = None
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
        # Reshape the data to be 3D for CNN
        trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
        testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))

        return trainX, testX, trainY, testY

    def create_model(self,custom_model):
        """Create the CNN model. \n
        By default, the model has 2 convolutional layers, 1 max pooling layer, 1 flatten layer, 2 dense layers and 1 output layer. \n
        The model is compiled with the adam optimizer and the mean squared error loss function. \n
        However, you can define your own architecture by providing your own Sequential model architecture from keras. \n
        """
        if custom_model is not None and isinstance(custom_model, Sequential):
            model = custom_model
            input_shape = (self.trainX.shape[1], self.trainX.shape[2])
            if model.layers[0].input_shape[1:] != input_shape:
                raise ValueError("The input shape of the custom model does not match self.trainX shape.",input_shape)
        else :
            # Default CNN architecture
            model = Sequential()
            model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(self.trainX.shape[1], self.trainX.shape[2])))
            model.add(Conv1D(filters=32, kernel_size=1, activation='relu'))
            model.add(MaxPooling1D(pool_size=1))
            model.add(Flatten())
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
        """This function is used to train, test and forecast time series data using cnn neural networks. \n
         
          Returns : \n
          predictions (pandas series) : pandas series of the predictions \n
          test (pandas series) : pandas series of the test values \n
          train (pandas series) : pandas series of the train values \n         
        """

        # Fit the model to the training data
        history = self.model.fit(self.trainX, self.trainY, epochs=200, batch_size=30, validation_data=(self.testX, self.testY), shuffle=False,callbacks=[
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
        """This function is used to forecast time series data using CNN neural networks on new_data. \n
        
        Parameters : \n
        new_data (numpy array) : numpy array of the new data to forecast \n
        
        Returns : \n
        predictions (numpy array) : numpy array of the predictions \n
        """
        print("new_data",new_data)
        print("new_data.shape",new_data.shape)
        print("self.trainX.shape",self.trainX.shape)

        # Reshape the data to be 3D for CNN
        new_data = np.reshape(new_data, (new_data.shape[1], new_data.shape[2], 1))
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
    model = cnn(df = df, target = "Close", test_size = 0.2, plot_learning_curves_bool = False)

    test = Sequential()
    test.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(3, 1)))
    test.add(Conv1D(filters=32, kernel_size=1, activation='relu'))
    test.add(MaxPooling1D(pool_size=1))
    test.add(Flatten())
    test.add(Dense(1, activation='linear'))
    test.compile(loss='mse', optimizer='adam')
    #model.create_model(test)
    #Forecast the time series data
    test=model.fit()
    forecast = model.forecast(new_data =np.array([[[2,3,4],[3,4,5]]]))
    print(forecast)
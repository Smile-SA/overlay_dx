from setuptools import setup

#with open('requirements.txt') as f:
#    requirements = f.read().splitlines()

setup(
    name='ts_forecaster_smile',
    version='0.1.19',
    author='Thomas Jaillon',
    author_email='thomas.jaillon@smile.fr',
    description='ts_forecaster is a library designed to forecast time series data. Using different models and algorithms such as XGBoost, LSTM, and SARIMA as well as ensemble learning methods and evaluation metrics',
    packages=['src/models', 'src/processing_forecasts'],
    install_requires=["matplotlib",
                      "numpy",
                      "pandas",
                      "protobuf",
                      "scikit-learn",
                      "seaborn",
                      "statsmodels",
                      "xgboost",
                      "tensorflow",
                      "keras",
                      "platypus-opt",
                      "scikit-learn"],
)
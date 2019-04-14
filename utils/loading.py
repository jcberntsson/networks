# (c) 2019 Joakim Berntsson
# Loading of data.
import csv
import pandas as pd

import tensorflow as tf
import numpy as np


class Loader:
    """Data loading class with support for:
    * loading csv files using pandas
    * data curation
    * column selection
    * creating train/test split with normalization.
    Note that this class only supports continuous values.
    """

    def __init__(self, filepath):
        self._data = pd.read_csv(filepath, header=0)

    def get_data(self):
        """Get the underlying pandas dataframe."""
        return self._data

    def apply(self, column_name, func):
        """Apply a function to a specific column to curate data."""
        self._data = self._data.dropna(subset=[column_name])
        self._data[column_name] = self._data[column_name].apply(func)

    def set_features(self, features: list):
        """Limit the loader to certain columns."""
        self._data = self._data[features]

    def get_data_split(self, train_fraction: float, target_name: str = 'target'):
        """Split the underlying dataframe into train/test sets. Note that
        the returned objects are TensorSliceDatasets."""
        features = self._data.columns.difference([target_name])

        # Divide into train and test
        train = self._data.sample(frac=train_fraction, random_state=200)
        test = self._data.drop(train.index)
    
        train_data = train[features].values.astype(float)
        train_labels = np.expand_dims(train[target_name].values, axis=1)
        test_data = test[features].values.astype(float)
        test_labels = np.expand_dims(test[target_name].values, axis=1)

        # Normalize
        train_mean = train_data.mean(axis=0)
        train_std = train_data.std(axis=0)
        train_data = (train_data - train_mean) / train_std
        test_data = (test_data - train_mean) / train_std

        # Construct tensorflow datasets
        train_dataset = tf.data.Dataset.from_tensor_slices((
            tf.cast(train_data, tf.float32),
            tf.cast(train_labels, tf.float32)))
        test_dataset = tf.data.Dataset.from_tensor_slices((
            tf.cast(test_data, tf.float32),
            tf.cast(test_labels, tf.float32)))
        
        return train_dataset, test_dataset

    def get_features(self):
        """Get the features for the data."""
        return self._data.columns

"""
-----------
DESCRIPTION
-----------
This is a simple feed forward network for heart disease dataset
    https://www.kaggle.com/ronitf/heart-disease-uci

-----------
PERFORMANCE
-----------
This model achieves ~75% test accuracy after 20 epochs using a 50/50
split for training and testing.

--------
TRAINING
--------
The network consists of two layers with 128 units each, with relu activation, 
and batch normalization. The final layer uses sigmoid activation.
During training the batch size was set to 32, and Adam optimizer with 0.001 learning rate.
"""
import csv
import pandas as pd

import tensorflow as tf
import numpy as np

tf.enable_eager_execution()

from tensorflow.keras.layers import Dense, BatchNormalization, Dropout

# Network constants
EPOCHS = 20
LEARNING_RATE = 0.001
BATCH_SIZE = 32
TRAIN_FRACTION = 0.5

class Network(tf.keras.Model):
    """Simple classification network using fully connected layers."""

    def __init__(self):
        super(Network, self).__init__()
    
        self.fc1 = Dense(128, activation="relu", name="fc1")
        self.bnorm1 = BatchNormalization(name="bnorm1")
        self.dropout1 = Dropout(0.1, name="dropout")
        self.fc2 = Dense(128, activation="relu", name="fc2")
        self.bnorm2 = BatchNormalization(name="bnorm2")
        self.final = Dense(1, name="final")

    def call(self, inputs, training=True):
        """Forward propagation for the network."""

        with tf.variable_scope("simple_network"):
            intermed = self.fc1(inputs)
            intermed = self.bnorm1(intermed, training=training)
            intermed = self.dropout1(intermed)
            intermed = self.fc2(intermed)
            intermed = self.bnorm2(intermed, training=training)
            intermed = self.final(intermed)
        return intermed

def prepare_dataset(df, features):
    # Divide into train and test
    train = df.sample(frac=TRAIN_FRACTION, random_state=200)
    test = df.drop(train.index)

    train_data = train[features].values
    train_labels = train['target'].values
    test_data = test[features].values
    test_labels = test['target'].values

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

def compute_accuracy(preds, targets):
    corrects = tf.cast(tf.equal(preds, targets), tf.float32)
    return tf.reduce_mean(corrects)

def main():
    # Load data
    hearts_data = pd.read_csv('heart.csv', header=0)

    # Log data info
    print(list(hearts_data))
    print(hearts_data.describe())
    print("Size: {}".format(hearts_data.shape))

    # Prepare dataset - use all features except target
    features = hearts_data.columns.difference(['target'])
    train_data, test_data = prepare_dataset(hearts_data, features)

    # Create model and optimizer
    model = Network()
    optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)

    # Progress saving
    all_accuracies = []
    all_losses = []

    # Train model on train split
    for i in range(EPOCHS):
        epoch_losses = []
        epoch_accuracies = []

        for inputs, targets in train_data.batch(BATCH_SIZE):
            targets = tf.expand_dims(targets, axis=1)

            with tf.GradientTape() as tape:
                # Forward prop
                logits = model(inputs)

                # Loss
                losses = tf.losses.sigmoid_cross_entropy(targets, logits)
                loss_value = tf.reduce_mean(losses)
            epoch_losses.append(loss_value)

            # Predict
            preds = tf.round(tf.sigmoid(logits))

            # Backward prop
            grads = tape.gradient(loss_value, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            epoch_accuracies.append(compute_accuracy(preds, targets))
        
        # Logging and saving states
        epoch_loss = np.mean(epoch_losses)
        epoch_accuracy = np.mean(epoch_accuracies)
        print("Epoch {:0>2d} - Loss = {:.4f}, Accuracy = {:.4f}".format(i, epoch_loss, epoch_accuracy))
        all_accuracies.append(epoch_accuracy)
        all_losses.append(epoch_loss)

    # Evaluate model on test split
    eval_accuracies = []
    for inputs, targets in test_data.batch(BATCH_SIZE):
        logits = model(inputs, training=False)
        preds = tf.round(tf.sigmoid(logits))
        acc = compute_accuracy(preds, targets)
        eval_accuracies.append(acc)
    print("Test set accuracy = {}".format(np.mean(eval_accuracies)))

    # Plot results
    import matplotlib.pyplot as plt
    _, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
    ax1.plot(all_accuracies)
    ax1.set_title("Accuracy")
    ax2.plot(all_losses)
    ax2.set_title("Loss")
    plt.show()

if __name__ == "__main__":
    main()

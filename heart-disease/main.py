"""
-----------
DESCRIPTION
-----------
This is a simple feed forward network for heart disease dataset
    https://www.kaggle.com/ronitf/heart-disease-uci

-----------
PERFORMANCE
-----------
This model achieves ~80% test accuracy after 500 epochs using a 80/20
split for training and testing.

--------
TRAINING
--------
The network consists of two layers with 32 units each, with relu activation, 
and l2 regularization. The final layer uses sigmoid activation.
During training the batch size was set to 64, and RMSProp with 0.0005 learning rate.
"""
import csv
import pandas as pd

import tensorflow as tf
import numpy as np

tf.enable_eager_execution()

from tensorflow.keras.layers import Dense, BatchNormalization, Dropout

# Network constants
EPOCHS = 500
LEARNING_RATE = 0.0005
BATCH_SIZE = 64
TRAIN_FRACTION = 0.8
WEIGHT_DECAY = 0.02
DROPOUT = 0.1

class Network(tf.keras.Model):
    """Simple classification network using fully connected layers."""

    def __init__(self, units=[16, 16], use_reg=True, use_bn=True, use_dropout=True):
        super(Network, self).__init__()

        self._net = []

        for i, unit in enumerate(units):
            reg = None
            if use_reg:
                reg = tf.contrib.layers.l2_regularizer(scale=WEIGHT_DECAY)
            fc = Dense(unit, kernel_regularizer=reg, name="fc{}".format(i))
            self._net.append(fc)
            if use_bn:
                bn = BatchNormalization(name="bnorm{}".format(i))
                self._net.append(bn)
            if use_dropout:
                do = Dropout(DROPOUT, name="dropout{}".format(i))
                self._net.append(do)
            self._net.append(tf.keras.activations.relu)
        
        regf = None
        if use_reg:
            regf = tf.contrib.layers.l2_regularizer(scale=WEIGHT_DECAY)
        final = Dense(1, kernel_regularizer=regf, name="final")
        self._net.append(final)
        self._net.append(tf.keras.activations.sigmoid)

    @property
    def trainable_variables(self):
        variables = []
        for comp in self._net:
            if hasattr(comp, "trainable_weights"):
                variables += comp.trainable_weights
        return variables

    def call(self, inputs, training=True):
        """Forward propagation for the network."""
        with tf.variable_scope("simple_network"):
            intermed = inputs
            for comp in self._net:
                if isinstance(comp, BatchNormalization):
                    intermed = comp(intermed, training=training)
                else:
                    intermed = comp(intermed)
        return intermed

    def losses(self):
        losses = []
        for comp in self._net:
            if hasattr(comp, "losses"):
                losses += comp.losses
        return tf.add_n(losses)

    def total_loss(self, targets, logits):
        losses = tf.losses.absolute_difference(targets, logits)
        loss_value = tf.reduce_mean(losses)
        loss_value += self.losses()
        return loss_value


def prepare_dataset(df, features):
    # Divide into train and test
    train = df.sample(frac=TRAIN_FRACTION, random_state=200)
    test = df.drop(train.index)

    train_data = train[features].values
    train_labels = np.expand_dims(train['target'].values, axis=1)
    test_data = test[features].values
    test_labels = np.expand_dims(test['target'].values, axis=1)

    # Normalize
    train_mean = train_data.mean(axis=0)
    train_std = train_data.std(axis=0)
    train_data = (train_data - train_mean) / train_std
    test_data = (test_data - train_mean) / train_std

    print("Training data size: {}\nTest data size: {}".format(train_data.shape, test_data.shape))

    # Construct tensorflow datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((
        tf.cast(train_data, tf.float32),
        tf.cast(train_labels, tf.float32)))
    test_dataset = tf.data.Dataset.from_tensor_slices((
        tf.cast(test_data, tf.float32),
        tf.cast(test_labels, tf.float32)))
    
    return train_dataset, test_dataset

def compute_accuracy(logits, targets):
    preds = tf.round(logits)
    corrects = 1 - tf.abs(preds - targets)
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
    model = Network(units=[32, 32], use_dropout=False, use_bn=False, use_reg=True)
    optimizer = tf.train.RMSPropOptimizer(learning_rate=LEARNING_RATE)

    # Progress saving
    all_accuracies = []
    all_losses = []
    test_accuracies = []
    test_losses = []

    # Train model on train split
    for i in range(EPOCHS):
        epoch_losses = []
        epoch_accuracies = []

        for inputs, targets in train_data.batch(BATCH_SIZE):

            with tf.GradientTape() as tape:
                # Forward prop
                logits = model(inputs)

                # Loss
                loss_value = model.total_loss(targets, logits)
            epoch_losses.append(loss_value)

            # Backward prop
            grads = tape.gradient(loss_value, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            epoch_accuracies.append(compute_accuracy(logits, targets))
        
        # Logging and saving states
        epoch_loss = np.mean(epoch_losses)
        epoch_accuracy = np.mean(epoch_accuracies)
        all_accuracies.append(epoch_accuracy)
        all_losses.append(epoch_loss)

        # Evaluate model on test split
        epoch_test_losses = []
        epoch_test_accuracies = []
        for inputs, targets in test_data.batch(BATCH_SIZE):
            logits = model(inputs, training=False)
            loss_value = model.total_loss(targets, logits)
            acc = compute_accuracy(logits, targets)

            epoch_test_losses.append(loss_value)
            epoch_test_accuracies.append(acc)
        epoch_test_loss = np.mean(epoch_test_losses)
        epoch_test_accuarcy = np.mean(epoch_test_accuracies)
        test_accuracies.append(epoch_test_accuarcy)
        test_losses.append(epoch_test_loss)

        print("Epoch {:0>2d} - Training[Loss = {:.4f}, Acc = {:.4f}], Test[Loss = {:.4f}, Acc = {:.4f}]"\
            .format(i, epoch_loss, epoch_accuracy, epoch_test_loss, epoch_test_accuarcy))

    # Plot results
    import matplotlib.pyplot as plt
    _, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)
    ax1.plot(all_accuracies)
    ax1.plot(test_accuracies)
    ax1.set_title("Accuracy")
    ax1.legend(['Training', 'Testing'])
    ax2.plot(all_losses)
    ax2.plot(test_losses)
    ax2.set_title("Loss")
    ax2.legend(['Training', 'Testing'])
    plt.show()

if __name__ == "__main__":
    main()

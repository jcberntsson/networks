"""
TODO: This model is in progress...
-----------
DESCRIPTION
-----------
This is a simple feed forward network for fifa dataset
    https://www.kaggle.com/karangadiya/fifa19

-----------
PERFORMANCE
-----------
TODO

--------
TRAINING
--------
TODO
"""
import tensorflow as tf
import numpy as np

from models.mlp import FeedForwardNetwork
from utils.loading import Loader

tf.enable_eager_execution()

from tensorflow.keras.layers import Dense, BatchNormalization, Dropout

# Network constants
EPOCHS = 20
LEARNING_RATE = 0.001
BATCH_SIZE = 64
TRAIN_FRACTION = 0.8
WEIGHT_DECAY = 0.02
DROPOUT = 0.1

def compute_accuracy(logits, targets):
    losses = tf.losses.mean_squared_error(targets, logits)
    return tf.reduce_mean(losses)

def height_conversion(feet_and_inches):
    feet, inches = feet_and_inches.split("'")
    return float(feet) * 30.48 + float(inches) * 2.54

def weight_conversion(lbs):
    lbs = lbs[:-3]
    lbs = float(lbs)
    return lbs * 0.45359237

def total_loss(model, targets, logits):
    losses = tf.losses.huber_loss(targets, logits)
    loss_value = tf.reduce_mean(losses)
    loss_value += model.losses()
    return loss_value

def main():
    # Load data
    loader = Loader('datasets/fifa.csv')

    # Explore data
    dd = loader.get_data()
    print(list(dd))
    print(dd.describe())
    print("Size: {}".format(dd.shape))

    # Create data split
    features = ['Age', 'Height', 'Weight']
    loader.apply('Height', height_conversion)
    loader.apply('Weight', weight_conversion)
    train_data, test_data = loader.get_data_split(features, TRAIN_FRACTION, target_name='Agility')

    # Create model and optimizer
    model = FeedForwardNetwork(units=[16, 16], use_sigmoid=False)
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
                loss_value = total_loss(model, targets, logits)
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
            loss_value = total_loss(model, targets, logits)
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
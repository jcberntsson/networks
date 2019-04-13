# (c) 2019 Joakim Berntsson
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
import tensorflow as tf
import numpy as np

from models.mlp import FeedForwardNetwork
from utils.loading import Loader
from utils.trainer import Trainer

tf.enable_eager_execution()

# Network constants
EPOCHS = 500
LEARNING_RATE = 0.0005
BATCH_SIZE = 64
TRAIN_FRACTION = 0.8
WEIGHT_DECAY = 0.02
DROPOUT = 0.1

def compute_accuracy(logits, targets):
    preds = tf.round(logits)
    corrects = 1 - tf.abs(preds - targets)
    return tf.reduce_mean(corrects)

def compute_loss(targets, logits):
    losses = tf.losses.absolute_difference(targets, logits)
    loss_value = tf.reduce_mean(losses)
    return loss_value

def main():
    # Load data
    loader = Loader('datasets/heart.csv')
    features = loader.get_features().difference(['target'])
    train_data, test_data = loader.get_data_split(features, TRAIN_FRACTION)

    # Create model and optimizer
    model = FeedForwardNetwork(units=[32, 32], l2_weight_decay=WEIGHT_DECAY)
    optimizer = tf.train.RMSPropOptimizer(learning_rate=LEARNING_RATE)

    # Train
    trainer = Trainer(model, optimizer, train_data, test_data, compute_loss, compute_accuracy)
    train_accs, train_losses, test_accs, test_losses = trainer.train(EPOCHS, BATCH_SIZE)

    # Plot results
    import matplotlib.pyplot as plt
    _, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)
    ax1.plot(train_accs)
    ax1.plot(test_accs)
    ax1.set_title("Accuracy")
    ax1.legend(['Training', 'Testing'])
    ax2.plot(train_losses)
    ax2.plot(test_losses)
    ax2.set_title("Loss")
    ax2.legend(['Training', 'Testing'])
    plt.show()

if __name__ == "__main__":
    main()

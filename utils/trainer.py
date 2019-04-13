# (c) 2019 Joakim Berntsson
# Trainer class
import tensorflow as tf
import numpy as np

class Trainer:
    """Trainer class for tensorflow"""

    def __init__(self, model, optimizer, train_data, test_data, compute_loss, compute_accuracy):
        self._model = model
        self._optimizer = optimizer
        self._train_data = train_data
        self._test_data = test_data
        self._compute_loss = compute_loss
        self._compute_accuracy = compute_accuracy

    def train(self, n_epochs: int, batch_size: int):
        # Progress saving
        all_accuracies = []
        all_losses = []
        test_accuracies = []
        test_losses = []

        # Train model on train split
        for i in range(n_epochs):
            epoch_losses = []
            epoch_accuracies = []

            for inputs, targets in self._train_data.batch(batch_size):

                with tf.GradientTape() as tape:
                    # Forward prop
                    logits = self._model(inputs)

                    # Loss
                    loss_value = self._model.regularize_loss() + self._compute_loss(targets, logits)
                epoch_losses.append(loss_value)

                # Backward prop
                grads = tape.gradient(loss_value, self._model.trainable_variables)
                self._optimizer.apply_gradients(zip(grads, self._model.trainable_variables))

                epoch_accuracies.append(self._compute_accuracy(logits, targets))
            
            # Logging and saving states
            epoch_loss = np.mean(epoch_losses)
            epoch_accuracy = np.mean(epoch_accuracies)
            all_accuracies.append(epoch_accuracy)
            all_losses.append(epoch_loss)

            # Evaluate model on test split
            epoch_test_losses = []
            epoch_test_accuracies = []
            for inputs, targets in self._test_data.batch(batch_size):
                logits = self._model(inputs, training=False)
                loss_value = self._model.regularize_loss() + self._compute_loss(targets, logits)
                acc = self._compute_accuracy(logits, targets)

                epoch_test_losses.append(loss_value)
                epoch_test_accuracies.append(acc)
            epoch_test_loss = np.mean(epoch_test_losses)
            epoch_test_accuarcy = np.mean(epoch_test_accuracies)
            test_accuracies.append(epoch_test_accuarcy)
            test_losses.append(epoch_test_loss)

            # Print progress
            print("Epoch {:0>2d} - Train[Loss={:.3f}, Metric={:.3f}], Test[Loss={:.3f}, Metric={:.3f}]"\
                .format(i+1, epoch_loss, epoch_accuracy, epoch_test_loss, epoch_test_accuarcy))

        return all_losses, all_accuracies, test_losses, test_accuracies

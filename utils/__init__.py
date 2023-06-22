import numpy as np
from deepmeg.training.callbacks import Callback
from deepmeg.training.trainers import Trainer
from copy import deepcopy
import torch

def balance(X: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    classes, classes_samples = np.unique(Y, return_counts=True)
    smallest_class = classes[np.argsort(classes_samples)][0]
    samples = classes_samples.min()
    X_list, Y_list = list(), list()
    stat = {class_: 0 for class_ in classes}

    for x, y in zip(X, Y):
        if y != smallest_class and stat[y] >= samples:
            continue
        else:
            Y_list.append(y)
            X_list.append(x)
            stat[y] += 1

    return np.array(X_list), np.array(Y_list)


def accuracy(predictions, targets):
    """
    Computes the accuracy metric for multiclass tasks with one-hot encoded labels.

    Args:
    - predictions: A PyTorch tensor of shape (batch_size, num_classes) representing the model's predictions.
    - targets: A PyTorch tensor of shape (batch_size, num_classes) representing the true labels in one-hot encoded format.

    Returns:
    - accuracy: A float value representing the accuracy metric.
    """

    # Convert one-hot encoded labels to class indices
    targets = torch.argmax(targets, dim=1)

    # Compute predicted class indices
    _, predicted = torch.max(predictions, dim=1)

    # Compute accuracy
    correct = torch.sum(predicted == targets)
    total = targets.shape[0]
    accuracy = correct / total

    return accuracy


class R2Score(torch.nn.Module):
    def __init__(self):
        super(R2Score, self).__init__()

    def forward(self, y_pred, y_true):
        total_sum_of_squares = torch.sum((y_true - torch.mean(y_true))**2)
        residual_sum_of_squares = torch.sum((y_true - y_pred)**2)
        r2 = 1 - residual_sum_of_squares / (total_sum_of_squares + 1e-7)  # Add a small constant to avoid division by zero
        return r2


class PenalizedEarlyStopping(Callback):
    def __init__(self, patience=5, monitor='loss_train', measure='binary_accuracy_train', min_delta=0, restore_best_weights=True):
        super().__init__()
        self.patience = patience
        self.monitor = monitor
        self.measure = measure
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.counter = 0
        self.min_criterion_value = np.inf
        self.max_measure_value = -np.inf
        self.best_weights = None

    def set_trainer(self, trainer: Trainer):
        super().set_trainer(trainer)
        self.model = self.trainer.model

    def on_epoch_end(self, epoch_num, metrics):
        criterion_value = metrics[self.monitor]
        measure_value = metrics[self.measure]
        if criterion_value < self.min_criterion_value or measure_value > self.max_measure_value:
            self.min_criterion_value = criterion_value
            self.counter = 0

            if measure_value > self.max_measure_value:
                self.best_weights = deepcopy(self.model.state_dict())
                self.max_measure_value = measure_value

        elif criterion_value > (self.min_criterion_value + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                if self.restore_best_weights:
                    self.restore()

                self.trainer.interrupt()

    def restore(self):
        self.model.load_state_dict(self.best_weights)


class WeightClipper(object):
    def __init__(self, model):
        self.model = model

    def __call__(self, module):
        self.model.temp_conv.weight -= self.model.temp_conv.weight.mean(-1, keepdim=True)


class TempConvAveClipping(Callback):
    def __init__(self):
        super().__init__()

    def set_trainer(self, trainer: Trainer):
        super().set_trainer(trainer)
        self.model = self.trainer.model
        self.clipper = WeightClipper(self.model)

    @torch.no_grad()
    def on_batch_end(
        self,
        Y: torch.Tensor,
        Y_pred: torch.Tensor,
        metrics: dict
    ):
        self.model.apply(self.clipper)


class IndependanceConstraint(Callback):
    def __init__(self, n_latent: int):
        self.n_latent = n_latent
        self.eye = torch.eye(self.n_latent)
        super().__init__()

    def set_trainer(self, trainer: 'Trainer'):
        super().set_trainer(trainer)
        self.model = self.trainer.model

    def on_loss_computed(self, X, Y, Y_pred, loss, is_training=True):

        if is_training:
            x = self.model.unmixing_layer(X)

            x = x.permute(1, 0, -1)
            x = x.reshape(x.shape[0], x.shape[1]*x.shape[2])
            s = torch.corrcoef(x)
            err = (((s - self.eye)**2)/self.n_latent**2).sum()

            loss += err

        return loss

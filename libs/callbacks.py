import tensorflow as tf


class LRFind(tf.keras.callbacks.Callback):
    """Keras callback to find ideal learning rate"""

    def __init__(self, min_lr, max_lr, n_rounds, warmup):
        assert n_rounds > warmup, "n_rounds must be greater than warmup"
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.warmup = warmup
        self.n_rounds = n_rounds
        self.step_up = (max_lr / min_lr) ** (1 / (n_rounds - warmup))
        self.lrs = []
        self.losses = []

    def on_train_begin(self, logs=None):
        self.weights = self.model.get_weights()

    def on_train_batch_end(self, batch, logs=None):
        if len(self.losses) > self.warmup:
            self.model.optimizer.lr = self.model.optimizer.lr * self.step_up
        else:
            self.model.optimizer.lr = self.min_lr
        if self.model.optimizer.lr > self.max_lr:
            self.model.stop_training = True
        self.lrs.append(self.model.optimizer.lr.numpy())
        self.losses.append(logs["loss"])

    def on_train_end(self, logs=None):
        self.model.set_weights(self.weights)

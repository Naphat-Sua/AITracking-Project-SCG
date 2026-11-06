from tensorflow.keras.callbacks import Callback


class TrainingHistory(Callback):
    """Keras callback that records loss/accuracy (and validation metrics)
    at the end of every epoch, for live plotting."""

    def __init__(self):
        super().__init__()
        self.loss = []
        self.accuracy = []
        self.val_loss = []
        self.val_accuracy = []

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.loss.append(logs.get('loss'))
        self.accuracy.append(100 * logs.get('accuracy', 0))
        if 'val_loss' in logs:
            self.val_loss.append(logs.get('val_loss'))
            self.val_accuracy.append(100 * logs.get('val_accuracy', 0))

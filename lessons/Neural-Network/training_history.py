from tensorflow.keras.callbacks import Callback


class TrainingHistory(Callback):
    """Keras callback that records loss/accuracy at the end of every epoch."""

    def __init__(self):
        super().__init__()
        self.loss = []
        self.accuracy = []

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.loss.append(logs.get('loss'))
        self.accuracy.append(100 * logs.get('accuracy', 0))

import tflearn
class EarlyStopCallback(tflearn.callbacks.Callback):
    def __init__(self, max_epochs_without_improvement):
        self.max_epochs_without_improvement = max_epochs_without_improvement
        self.epochs_without_improvement = 0
        self.best_validation_loss = float('inf')
        
    def on_epoch_end(self, training_state):
        current_validation_loss = training_state.val_loss
        if current_validation_loss < self.best_validation_loss:
            self.best_validation_loss = current_validation_loss
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1
            if self.epochs_without_improvement >= self.max_epochs_without_improvement:
                raise StopIteration('Training stopped early due to lack of improvement in validation loss')

import torch

class EarlyStopping:
    def __init__(self,patience =10,delta=0.01):
        """
        Early stopping
        Args:
            patience: int, number of epochs to wait before stopping
            delta: float, the minimum improvements
        """
        self.patience = patience
        self.delta = delta
        self.counter =0 
        self.early_stop = False
        self.best_loss = float('inf')

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter =0 
        else:
            self.counter+=1
            if self.counter >= self.patience:
                self.early_stop = True
    


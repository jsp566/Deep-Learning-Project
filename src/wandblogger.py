import wandb


# config should be a dictionary containing hyperparameters and settings
# initializer
# number of layers
# nodes per layer
# loss function
# optimizer
# learning rate
# batch size
# epochs
# dataset



class Logger:
    def __init__(self, dataset="Deep-Learning-Project"):
        self.dataset = dataset

    def initialize(self, model, loss_function, optimizer):
        config = {
            "dataset": self.dataset,
            "model": type(model).__name__,
            "loss_function": type(loss_function).__name__,
            "optimizer": type(optimizer).__name__,
            "learning_rate": optimizer.learning_rate,
        }

        self.run = wandb.init(
            project=self.dataset,
            entity="DTU-Deep-Learning-Project",
            config=config,
        )
        self.run.define_metric("*", step_metric="epoch")

    def log_metrics(self, epoch, train_loss, val_loss):
        #train_loss = self.loss_function.compute(y_true, y_pred)
        metrics = {"epoch": epoch,
                   "train/loss": train_loss,
                   "val/loss": val_loss
                   }
        
        #for i, layer in enumerate(self.model.layers):
            

        
        self.run.log(metrics, step=epoch)

    def finish(self):
        self.run.finish()
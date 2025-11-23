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
    def __init__(self, project, config, model, loss_function):
        self.project = project
        self.config = config
        self.model = model
        self.loss_function = loss_function
        #self.run.define_metric("*", step_metric="epoch")

    def initialize(self):
        self.run = wandb.init(
            project=self.project,
            entity="DTU-Deep-Learning-Project",
            config=self.config,
        )

    def log_metrics(self, epoch, y_true, y_pred):
        #train_loss = self.loss_function.compute(y_true, y_pred)
        metrics = {"train/loss": None,
                   "train/accuracy": None,
                   "val/loss": None,
                   "val/accuracy": None
                   }
        
        for i, layer in enumerate(self.model.layers):
            metrics[f"weights/{str(i)}"] = layer.weights
            metrics[f"biases/{str(i)}"] = layer.biases
            metrics[f"grad_weights/{str(i)}"] = layer.grad_weights
            metrics[f"grad_biases/{str(i)}"] = layer.grad_biases

        
        self.run.log(metrics, step=epoch)

    def finish(self):
        self.run.finish()
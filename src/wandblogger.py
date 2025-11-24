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

    def initialize(self, model, loss_function, optimizer, epochs, batch_size):
        config = {
            "dataset": self.dataset,
            "model": type(model).__name__,
            "loss_function": type(loss_function).__name__,
            "optimizer": type(optimizer).__name__,
            "learning_rate": optimizer.learning_rate,
            "epochs": epochs,
            "batch_size": batch_size,
            }
        
        i = 0
        for layer in model.layers:
            if layer.__class__.__name__ == "Layer":
                i += 1
                config[f"layer_{i}/output_size"] = layer.output_size
                config[f"layer_{i}/weight_initializer"] = str(layer.weight_initializer)
                config[f"layer_{i}/bias_initializer"] = str(layer.bias_initializer)
                config[f"layer_{i}/Dropout_probability"] = 0.0
                config[f"layer_{i}/BatchNorm_epsilon"] = None
                config[f"layer_{i}/BatchNorm_momentum"] = None
            elif layer.__class__.__name__ == "Dropout":
                config[f"layer_{i}/Dropout_probability"] = layer.drop_probability
            elif layer.__class__.__name__ == "BatchNorm":
                config[f"layer_{i}/BatchNorm_epsilon"] = layer.epsilon
                config[f"layer_{i}/BatchNorm_momentum"] = layer.momentum

        config["layers"] = i
        self.run = wandb.init(
            project=self.dataset,
            entity="DTU-Deep-Learning-Project",
            config=config,
        )
        self.run.define_metric("*", step_metric="epoch")

    def log_metrics(self, epoch, train_loss, val_loss, model):
        #train_loss = self.loss_function.compute(y_true, y_pred)
        metrics = {"epoch": epoch,
                   "train/loss": train_loss,
                   "val/loss": val_loss,
                   }
        
        i = 0
        for layer in model.layers:
            if layer.__class__.__name__ == "Layer":
                i += 1
                metrics[f"layer_{i}/weights"] = wandb.Histogram(layer.weights)
                metrics[f"layer_{i}/biases"] = wandb.Histogram(layer.biases)
                metrics[f"layer_{i}/weight_gradients"] = wandb.Histogram(layer.dweights)
                metrics[f"layer_{i}/bias_gradients"] = wandb.Histogram(layer.dbiases)
            elif layer.__class__.__name__ == "BatchNorm":
                metrics[f"layer_{i}/gamma"] = wandb.Histogram(layer.gamma)
                metrics[f"layer_{i}/beta"] = wandb.Histogram(layer.beta)
                metrics[f"layer_{i}/gamma_gradients"] = wandb.Histogram(layer.dgamma)
                metrics[f"layer_{i}/beta_gradients"] = wandb.Histogram(layer.dbeta)

            

        
        self.run.log(metrics, step=epoch)

    def finish(self):
        self.run.finish()
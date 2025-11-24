import wandb
from src import preprocess, activation_functions, initializers, layer, loss_functions, optimizers, FFNN, training, wandblogger, dropout, batchnorm

def train_sweep(entity, project, config, x_train, y_train, x_valid, y_valid):
    wandb.init(entity=entity,
        project=project)        
    cfg = wandb.config  
    # building layers dynamically
    layers = []

    input_size = 28 * 28
    prev_size = input_size
    # hidden layers
    for hidden_size in cfg.layer_sizes:     
        layers.append(
            layer.Layer(
                input_size=prev_size,
                output_size=hidden_size,
                weight_initializer=initializers.HeInitializer(),
                bias_initializer=initializers.ConstantInitializer(0),
            )
        )
        # add batchnorm if specified
        if cfg.use_batchnorm:
            layers.append(batchnorm.BatchNorm(hidden_size))

        layers.append(activation_functions.ReLU())
        # add dropout if specified, had issues with 0.5 dropout one round
        if cfg.dropout_rate > 0:
            layers.append(dropout.Dropout(cfg.dropout_rate))

        prev_size = hidden_size

    # Output layer
    layers.append(
        layer.Layer(
            input_size=prev_size,
            output_size=10,
            weight_initializer=initializers.HeInitializer(),
            bias_initializer=initializers.ConstantInitializer(0),
        )
    )
    #add ADAM when it has been implemented
    if cfg.optimizer == "sgd":
        optimizer = optimizers.SGD(learning_rate=cfg.learning_rate)
    """elif cfg.optimizer == "adam":
        optimizer = optimizers.Adam(learning_rate=cfg.learning_rate)"""
    
    model = FFNN.FFNN(
        layers=layers,
        loss_function=loss_functions.CrossEntropyLoss(),
        optimizer=optimizer,
    )

    logger = wandblogger.Logger(project)
    trainer = training.Trainer(model=model, loss_function=loss_functions.CrossEntropyLoss(), optimizer=optimizer, logger=logger)

    history = trainer.train(
        x_train,
        y_train,x_val=x_valid,
        y_val=y_valid,
        early_stopper=training.EarlyStopping(),
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        shuffle=True
    )

    wandb.finish()

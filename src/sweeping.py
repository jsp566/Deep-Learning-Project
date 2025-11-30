import wandb
from src import (
    preprocess,
    activation_functions,
    initializers,
    layer,
    loss_functions,
    optimizers,
    FFNN,
    training,
    wandblogger,
    dropout,
    batchnorm,
    regularization,
)
import pickle
import os


def train_sweep(
    entity, project, config, x_train, y_train, x_valid, y_valid, input_size=28 * 28
):
    run = wandb.init(entity=entity, project=project)
    cfg = wandb.config
    # building layers dynamically
    if cfg.normalize_data:
        zScorenormalize = preprocess.ZScoreNormalize()
        x_train = zScorenormalize.transform(x_train)
        x_valid = zScorenormalize.transform(x_valid)
    layers = []
    if cfg.kernel_initializer == "HeInitializer()":
        kernel_initializer = initializers.HeInitializer()
    elif cfg.kernel_initializer == "GlorotInitializer()":
        kernel_initializer = initializers.GlorotInitializer()

    if cfg.activation_function == "ReLU()":
        activation_function = activation_functions.ReLU
    elif cfg.activation_function == "LeakyReLU()":
        activation_function = activation_functions.LeakyReLU
    elif cfg.activation_function == "Tanh()":
        activation_function = activation_functions.Tanh
    input_size = input_size
    prev_size = input_size
    
    # hidden layers
    layer_sizes = []

    try:
        layer_sizes = cfg.layer_sizes
    except AttributeError:
        layer_sizes = [cfg.n_hidden_units] * cfg.num_hidden_layers



    for hidden_size in layer_sizes:
        layers.append(
            layer.Layer(
                input_size=prev_size,
                output_size=hidden_size,
                weight_initializer=kernel_initializer,
                bias_initializer=initializers.ConstantInitializer(0),
            )
        )
        # add batchnorm if specified
        if cfg.use_batchnorm:
            layers.append(batchnorm.BatchNorm(hidden_size))

        layers.append(activation_function())
        # add dropout if specified, had issues with 0.5 dropout one round
        if cfg.dropout_rate > 0:
            layers.append(dropout.Dropout(cfg.dropout_rate))

        prev_size = hidden_size

    # Output layer
    layers.append(
        layer.Layer(
            input_size=prev_size,
            output_size=10,
            weight_initializer=kernel_initializer,
            bias_initializer=initializers.ConstantInitializer(0),
        )
    )
    # add ADAM when it has been implemented
    if cfg.optimizer == "sgd":
        optimizer = optimizers.SGD(learning_rate=cfg.learning_rate)
    elif cfg.optimizer == "adam":
        optimizer = optimizers.Adam(learning_rate=cfg.learning_rate)

    model = FFNN.FFNN(
        layers=layers,
    )

    logger = wandblogger.Logger(project)
    trainer = training.Trainer(
        model=model,
        loss_function=loss_functions.CrossEntropyLoss(),
        optimizer=optimizer,
        logger=logger,
        regularization=regularization.L2Regularization(cfg.l2_lambda),
    )

    history = trainer.train(
        x_train,
        y_train,
        x_val=x_valid,
        y_val=y_valid,
        early_stopper=training.EarlyStopping(),
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        shuffle=True,
    )
    filepath = os.path.join(run.dir, "model.pkl")

    with open(filepath, "wb") as output_file:
        pickle.dump(model, output_file)

    wandb.finish()

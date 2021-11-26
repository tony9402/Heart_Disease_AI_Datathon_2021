
class UNetConfig:
    resolution = 128
    batch_size = 4
    learning_rate = 0.001
    epoch = 20
    num_workers = 4
    gpus = [0]
    optimizer = {
        'name': 'RMSprop',
        'weight_decay': 1e-8,
        'momentum': 0.9
    }
    scheduler = {
        'name': 'ReduceLROnPlateau',
        'patience': 2
    }
    loss = 'CrossEntropyLoss'
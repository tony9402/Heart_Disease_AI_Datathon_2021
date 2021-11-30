from apex.optimizers import FusedAdam, FusedSGD
from timm.optim import AdamW
from torch import optim
from torch.optim import lr_scheduler
from torch.optim.rmsprop import RMSprop
from torch.optim.adamw import AdamW
from torch.optim.lr_scheduler import MultiStepLR, CyclicLR

from training.tools.scheduler import ExponentialLRScheduler, LRStepScheduler, PolyLR

def create_optimizer(optimizer_config, model):
    params = model.parameters()

    if optimizer_config.type == "SGD":
        optimizer = optim.SGD(params,
                              lr=optimizer_config.learning_rate,
                              momentum=optimizer_config.momentum,
                              weight_decay=optimizer_config.weight_decay,
                              nesterov=optimizer_config.nesterov)
    elif optimizer_config.type == "FusedSGD":
        optimizer = FusedSGD(params,
                             lr=optimizer_config.learning_rate,
                             momentum=optimizer_config.momentum,
                             weight_decay=optimizer_config.weight_decay,
                             nesterov=optimizer_config.nesterov)
    elif optimizer_config.type == "Adam":
        optimizer = optim.Adam(params,
                               lr=optimizer_config.learning_rate,
                               weight_decay=optimizer_config.weight_decay)
    elif optimizer_config.type == "FusedAdam":
        optimizer = FusedAdam(params,
                              lr=optimizer_config.learning_rate,
                              weight_decay=optimizer_config.weight_decay)
    elif optimizer_config.type == 'AdamW':
        optimizer = AdamW(params,
                          lr=optimizer_config.learning_rate,
                          weight_decay=optimizer_config.weight_decay)
    elif optimizer_config.type == "RmsProp":
        optimizer = RMSprop(params,
                            lr=optimizer_config.learning_rate,
                            weight_decay=optimizer_config.weight_decay)
    else:
        raise KeyError("unrecognized optimizer {}".format(optimizer_config.type))

    scheduler = None
    if optimizer_config.scheduler.type == "step":
        scheduler = LRStepScheduler(optimizer, **optimizer_config.scheduler.params)
    elif optimizer_config.scheduler.type == "clr":
        scheduler = CyclicLR(optimizer, **optimizer_config.scheduler.params)
    elif optimizer_config.scheduler.type == "multistep":
        scheduler = MultiStepLR(optimizer, **optimizer_config.scheduler.params)
    elif optimizer_config.scheduler.type == "exponential":
        scheduler = ExponentialLRScheduler(optimizer, **optimizer_config.scheduler.params)
    elif optimizer_config.scheduler.type == "poly":
        scheduler = PolyLR(optimizer, **optimizer_config.scheduler.params)
    elif optimizer_config.scheduler.type == "constant":
        scheduler = lr_scheduler.LambdaLR(optimizer, lambda epoch: 1.0)
    elif optimizer_config.scheduler.type == "cosine_lr":
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, **optimizer_config.scheduler.params)
    elif optimizer_config.scheduler.type == "cosine_warm":
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, **optimizer_config.scheduler.params)
    elif optimizer_config.scheduler.type == "linear":
        def linear_lr(it):
            return it * optimizer_config.scheduler.params.alpha + optimizer_config.scheduler.params.beta

        scheduler = lr_scheduler.LambdaLR(optimizer, linear_lr)

    return optimizer, scheduler
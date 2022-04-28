from dataclasses import dataclass


@dataclass()
class Utility:
    n_epoch:int = 30
    BATCH_SIZE:int = 100
    LR:float = 3e-4

    ## Image processing
    CHANNELS:int = 3
    IMAGE_SIZE_1:int = 224
    IMAGE_SIZE_2:int = 224

    # model selection
    model_name:str = "resnet18"
    
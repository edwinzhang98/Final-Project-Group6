from dataclasses import dataclass

@dataclass()
class Utility:
    n_epoch:int = 30
    BATCH_SIZE:int = 64
    LR:float = 0.001

    ## Image processing
    CHANNELS:int = 3
    IMAGE_SIZE_1:int = 224
    IMAGE_SIZE_2:int = 224

    ## model selection
    # model_name = "resnet18" 
    # model_name = "CNN" 
    model_name = "VIT" 

class Utility_CNN:
    n_epoch:int = 30
    BATCH_SIZE:int = 64
    LR:float = 0.001

    ## Image processing
    CHANNELS:int = 3
    IMAGE_SIZE_1:int = 224
    IMAGE_SIZE_2:int = 224

    ## model selection
    model_name = "CNN" 


class Utility_RESNET18:
    n_epoch:int = 30
    BATCH_SIZE:int = 64
    LR:float = 0.003

    ## Image processing
    CHANNELS:int = 3
    IMAGE_SIZE_1:int = 224
    IMAGE_SIZE_2:int = 224

    ## model selection
    model_name = "resnet18"

class Utility_VIT:
    n_epoch:int = 30
    BATCH_SIZE:int = 64
    LR:float = 0.001

    ## Image processing
    CHANNELS:int = 3
    IMAGE_SIZE_1:int = 224
    IMAGE_SIZE_2:int = 224

    ## model selection
    model_name = "VIT"

    
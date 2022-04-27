from dataclasses import dataclass

@dataclass()
class Utility:
    n_epoch:int = 20
    BATCH_SIZE:int = 50
    LR:float = 3e-4 #0.001

    ## Image processing
    CHANNELS:int = 3
    IMAGE_SIZE_1:int = 224
    IMAGE_SIZE_2:int = 224
    NICKNAME: str = "hui_cnn"

    
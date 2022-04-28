# ------------------------------------------------------------------------------------------------------------------
from sklearn.preprocessing import LabelEncoder
import cv2
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, hamming_loss, \
    cohen_kappa_score, matthews_corrcoef
import torch
import torch.nn as nn
import numpy as np
from torch.utils import data
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchvision.transforms as transforms
from torchvision import models
from tqdm import tqdm
import os
from typing import Dict, List, Tuple, Union, Any
from utility import Utility
from sklearn.model_selection import train_test_split
from torchsampler import ImbalancedDatasetSampler

# ------------------------------------------------------------------------------------------------------------------


class CNN(nn.Module):

    def __init__(self, n_class:int):
        super().__init__()
        self.conv1:nn.Conv2d = nn.Conv2d(3, 16, (3, 3))
        self.convnorm1:nn.BatchNorm2d = nn.BatchNorm2d(16)
        self.pad1:nn.ZeroPad2d = nn.ZeroPad2d(2)

        self.conv2:nn.Conv2d = nn.Conv2d(16, 32, (3, 3))
        self.convnorm2:nn.BatchNorm2d = nn.BatchNorm2d(32)
        self.pool2:nn.MaxPool2d = nn.MaxPool2d((2, 2))

        self.conv3:nn.Conv2d = nn.Conv2d(32, 64, (3, 3))
        self.convnorm3:nn.BatchNorm2d = nn.BatchNorm2d(64)

        self.conv4:nn.Conv2d = nn.Conv2d(64, 128, (3, 3))
        self.global_avg_pool:nn.AdaptiveAvgPool2d = nn.AdaptiveAvgPool2d((1, 1))

        self.linear:nn.Linear = nn.Linear(128, n_class)
        self.act:nn.ReLU = nn.ReLU()
        self.softmax:nn.Softmax = nn.Softmax(dim = 1)
       
    def forward(self, input:torch.Tensor):
        conv_layer1_out = self.pad1(self.convnorm1(self.act(self.conv1(input))))
        conv_layer2_out = self.pool2(self.convnorm2(self.act(self.conv2(conv_layer1_out))))
        conv_layer3_out = self.act(self.conv4(self.convnorm3(self.act(self.conv3(conv_layer2_out)))))
        y = self.linear(self.global_avg_pool(conv_layer3_out).view(-1, 128))
        return y


# ------------------------------------------------------------------------------------------------------------------


class Custom_Dataset(data.Dataset):
    '''
    From : https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel
    '''
    def __init__(
            self,
            list_IDs:List[Any],
            type_data:str,
            target_type:int,
            IMAGE_SIZE_1:int,
            IMAGE_SIZE_2:int,
            DATA_DIR:str,
            xdf_dset_train:pd.DataFrame,
            xdf_dset_valid:pd.DataFrame,
            xdf_dset_test:pd.DataFrame,
            n_class:int) -> None:
        
        super().__init__()

        # Initialization'
        self.list_IDs:List[Any] = list_IDs
        self.type_data:str = type_data
        self.target_type:int = target_type
        self.IMAGE_SIZE_1:int = IMAGE_SIZE_1
        self.IMAGE_SIZE_2:int = IMAGE_SIZE_2
        self.DATA_DIR:str = DATA_DIR
        self.xdf_dset_train:pd.DataFrame = xdf_dset_train
        self.xdf_dset_valid:pd.DataFrame = xdf_dset_valid
        self.xdf_dset_test:pd.DataFrame = xdf_dset_test
        self.n_class:int = n_class

        if self.type_data == "train":
            self.transform = transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.Resize((self.IMAGE_SIZE_1, self.IMAGE_SIZE_2)),
                    # transforms.RandomHorizontalFlip(p=0.5),
                    # transforms.RandomRotation(degrees=45),
                    transforms.ToTensor()
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.Resize((self.IMAGE_SIZE_1, self.IMAGE_SIZE_2)),
                    transforms.ToTensor(),
                ]
            )

    def __len__(self):
        # Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index:int):
        # Generates one sample of data'
        # Select sample
        ID:int = self.list_IDs[index]
        # Load data and get label
        
        if self.type_data == 'train':
            file:str = self.DATA_DIR\
                + self.xdf_dset_train.loc[ID, "image"].replace("images/", "")
            t:Union[str, int] = self.xdf_dset_train.loc[ID, "target_class"]
        elif self.type_data == "valid":
            file = self.DATA_DIR\
                + self.xdf_dset_valid.loc[ID, "image"].replace("images/", "")
            t = self.xdf_dset_valid.loc[ID, "target_class"]
        else:
            file = self.DATA_DIR\
                + self.xdf_dset_test.loc[ID, "image"].replace("images/", "")


        if self.target_type == 2:
            t_lst:List[str] = t.split(",")
            labels_ml:List[int] = [int(e) for e in t_lst]
            t_tensor:torch.Tensor = torch.FloatTensor(labels_ml)
            return t_tensor


        img:np.ndarray = cv2.imread(file)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        img:torch.Tensor = self.transform(img)
        
        img = torch.FloatTensor(img)
        
        
        # Augmentation only for train
        
        return img, t

    # def get_labels(self):
    #     if self.type_data == 'train':
    #         t:Union[str, int] = self.xdf_dset_train["target_class"]
    #     elif self.type_data == "valid":
    #         t = self.xdf_dset_valid["target_class"]

    #     return t
# ------------------------------------------------------------------------------------------------------------------


def read_data(
        BATCH_SIZE:int, 
        IMAGE_SIZE_1:int, 
        IMAGE_SIZE_2:int, 
        DATA_DIR:str, 
        xdf_dset_train:pd.DataFrame,
        xdf_dset_valid:pd.DataFrame, 
        xdf_dset_test:pd.DataFrame, 
        n_class:int, 
        target_type:int):

    # ---------------------- Parameters for the data loader --------------------------------

    list_of_ids_train:List[pd.RangeIndex] = list(xdf_dset_train.index)
    list_of_ids_valid:List[pd.RangeIndex] = list(xdf_dset_valid.index)
    list_of_ids_test:List[pd.RangeIndex] = list(xdf_dset_test.index)

    # Data Loaders

    params:Dict[str, Union[int, bool]] = {
        'batch_size': BATCH_SIZE,
        "shuffle": True
    }

    training_set:Custom_Dataset = Custom_Dataset(
        list_of_ids_train, 
        'train', 
        target_type, 
        IMAGE_SIZE_1, 
        IMAGE_SIZE_2, 
        DATA_DIR, 
        xdf_dset_train, 
        xdf_dset_valid,
        xdf_dset_test, 
        n_class
    )

    training_generator:data.DataLoader = data.DataLoader(training_set, **params)

    params:Dict[str, Union[int, bool]] = {
        'batch_size': BATCH_SIZE,
        'shuffle': False
    }

    validation_set:Custom_Dataset = Custom_Dataset(
        list_of_ids_valid, 
        'valid', 
        target_type, 
        IMAGE_SIZE_1, 
        IMAGE_SIZE_2, 
        DATA_DIR, 
        xdf_dset_train,
        xdf_dset_valid, 
        xdf_dset_test, 
        n_class
    )

    validation_generator:data.DataLoader = data.DataLoader(validation_set, **params)

    # Make the channel as a list to make it variable

    return training_generator, validation_generator
# ------------------------------------------------------------------------------------------------------------------


def save_model(model, OR_PATH) -> None: 
    '''
      Print Model Summary
    '''

    print(model, file=open(OR_PATH +os.path.sep + "models" + os.path.sep
                + 'summary.txt', "w"))
# ------------------------------------------------------------------------------------------------------------------


def model_definition(
        LR:float,
        device:str, 
        n_class:int,
        OR_PATH:str, 
        model_name:str): 
    '''
        Define a Keras sequential model
        Compile the model
    '''

    if model_name == "resnet18":
        model:models.ResNet = models.resnet18(pretrained = True)
        model.fc = nn.Linear(model.fc.in_features, n_class)
    elif model_name == "resnet50":
        model:models.ResNet = models.resnet50(pretrained = True)
        model.fc = nn.Linear(model.fc.in_features, n_class)
    elif model_name == "resnext50":
        model:models.ResNet = models.resnext50_32x4d(pretrained = True)
        model.fc = nn.Linear(model.fc.in_features, n_class)
    elif model_name == "vgg":
        model = models.vgg()
    elif model_name == "CNN":
        model:CNN = CNN(n_class)

    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-3)
    criterion:nn.CrossEntropyLoss = nn.CrossEntropyLoss()

    scheduler = ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.1, 
        patience=5, 
        verbose=True
    )

    save_model(model, OR_PATH)

    return model, optimizer, criterion, scheduler
# ------------------------------------------------------------------------------------------------------------------


def train_and_test(
        train_ds, 
        valid_ds, 
        list_of_metrics:List[str], 
        list_of_agg:List[str], 
        n_epoch:int, 
        LR:float,
        model_name:str,  
        device:str,   
        xdf_dset_valid:pd.DataFrame, 
        n_class:int, 
        OR_PATH:str,
        SAVE_MODEL:bool, 
        save_on:str):

    model, optimizer, criterion, scheduler = model_definition(
        LR, 
        device, 
        n_class,
        OR_PATH,
        model_name
    )

    cont:int = 0
    train_loss_item:List = []
    valid_loss_item:List = []
    total_valid_loss:List = []
    wrong_label:List = []
  

    model.phase = 0

    met_valid_best:float = 0
    for epoch in range(n_epoch):
        train_loss, steps_train = 0, 0

        model.train()
        pred_labels_per_hist:List[float] = []
        pred_logits:List[float] = []
        train_hist:List[float] = []
        valid_hist:List[float] = []

        with tqdm(total=len(train_ds), desc="Epoch {}".format(epoch)) as pbar:

            for xdata,xtarget in train_ds:

                xdata, xtarget = xdata.to(device), xtarget.to(device)
                optimizer.zero_grad()
                output = model(xdata)

                loss = criterion(output, xtarget)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                cont += 1

                steps_train += 1

                train_loss_item.append([epoch, loss.item()])

                pred_labels_per = output.detach().to(torch.device('cpu')).numpy()

                
                
                pred_labels_per_hist = pred_labels_per_hist + list(pred_labels_per)
                
                
                train_hist_per = xtarget.cpu().numpy()
                
            
                train_hist = train_hist + list(train_hist_per)

                pbar.update(1)
                pbar.set_postfix_str("Train Loss: {:.5f}".format(train_loss / steps_train))

              
                pred_logits = pred_logits + list(np.argmax(pred_labels_per, axis = 1))
                for (pred, true) in zip(pred_logits, train_hist):
                    if pred != true:
                        wrong_label.append(true)

                    

        

        # Metric Evaluation
        train_metrics = metrics_func(list_of_metrics, list_of_agg, train_hist, pred_logits)

        avg_train_loss = train_loss / steps_train

        ## Finish with Training

        ## Testing the model

        model.eval()
        
        pred_labels_per_hist_valid:List[float] = []

        pred_logits_valid:List[int] = []

        valid_loss, steps_valid = 0, 0
        met_valid = 0

        with torch.no_grad():

            with tqdm(total=len(valid_ds), desc="Epoch {}".format(epoch)) as pbar:

                for xdata,xtarget in valid_ds:

                    xdata, xtarget = xdata.to(device), xtarget.to(device)
                    optimizer.zero_grad()
                    output = model(xdata)

                    loss = criterion(output, xtarget)
                    valid_loss += loss.item()
                    cont += 1

                    steps_valid += 1

                    valid_loss_item.append([epoch, loss.item()])

                    pred_labels_per = output.detach().to(torch.device('cpu')).numpy()

                    pred_labels_per_hist = pred_labels_per_hist + list(pred_labels_per)
                    
                    valid_hist_per = xtarget.cpu().numpy()
                    
                    valid_hist = valid_hist + list(valid_hist_per)

                    pbar.update(1)
                    pbar.set_postfix_str("Validation Loss: {:.5f}".format(valid_loss / steps_valid))

                    pred_logits_valid = pred_logits_valid + list(np.argmax(pred_labels_per, axis = 1))
                    

        valid_metrics = metrics_func(list_of_metrics, list_of_agg, valid_hist, pred_logits_valid)

        avg_valid_loss = valid_loss / steps_valid
        
        scheduler.step(avg_valid_loss)
        total_valid_loss.append(avg_valid_loss)


        xstrres = "Epoch {}: ".format(epoch)
        for met, dat in train_metrics.items():
            xstrres = xstrres +' Train '+met+ ' {:.5f}'.format(dat)


        xstrres = xstrres + " - "
        for met, dat in valid_metrics.items():
            xstrres = xstrres + ' Validation '+met+ ' {:.5f}'.format(dat)
            if met == save_on:
                met_valid = dat

        print(xstrres)  #Print metrics

        if met_valid > met_valid_best and SAVE_MODEL:

            torch.save(model.state_dict(), OR_PATH +os.path.sep
                + "models" + os.path.sep + "model.pt")
            # xdf_dset_results = xdf_dset_valid.copy()

            # ## The following code creates a string to be saved as 1,2,3,3,
            # ## This code will be used to validate the model
            # xfinal_pred_labels = []
            # for i in range(len(pred_labels)):
            #     joined_string = ",".join(str(int(e)) for e in pred_labels[i])
            #     xfinal_pred_labels.append(joined_string)

            # xdf_dset_results['label'] = xfinal_pred_labels

            # xdf_dset_results.to_csv(OR_PATH +os.path.sep
            #     + "references" + os.path.sep + "test_copy.csv", mode = "a",
            #     header = ["label"], index = False)
            print("The model has been saved!")
            met_valid_best = met_valid
#------------------------------------------------------------------------------------------------------------------


def metrics_func(metrics, aggregates, y_true, y_pred):
    '''
    multiple functiosn of metrics to call each function
    f1, cohen, accuracy, mattews correlation
    list of metrics: f1_micro, f1_macro, f1_avg, coh, acc, mat
    list of aggregates : avg, sum
    :return:
    '''

    def f1_score_metric(y_true, y_pred, type):
        '''
        type = micro,macro,weighted,samples
        :param y_true:
        :param y_pred:
        :param average:
        :return: res
        '''
        res = f1_score(y_true, y_pred, average=type)
        return res

    def cohen_kappa_metric(y_true, y_pred):
        res = cohen_kappa_score(y_true, y_pred)
        return res

    def accuracy_metric(y_true, y_pred):
        res = accuracy_score(y_true, y_pred)
        return res

    def matthews_metric(y_true, y_pred):
        res = matthews_corrcoef(y_true, y_pred)
        return res

    def hamming_metric(y_true, y_pred):
        res = hamming_loss(y_true, y_pred)
        return res

    xcont = 1
    xsum = 0
    xavg = 0
    res_dict = {}
    for xm in metrics:
        if xm == 'f1_micro':
            # f1 score average = micro
            xmet = f1_score_metric(y_true, y_pred, 'micro')
        elif xm == 'f1_macro':
            # f1 score average = macro
            xmet = f1_score_metric(y_true, y_pred, 'macro')
        elif xm == 'f1_weighted':
            # f1 score average =
            xmet = f1_score_metric(y_true, y_pred, 'weighted')
        elif xm == 'coh':
             # Cohen kappa
            xmet = cohen_kappa_metric(y_true, y_pred)
        elif xm == 'acc':
            # Accuracy
            xmet =accuracy_metric(y_true, y_pred)
        elif xm == 'mat':
            # Matthews
            xmet =matthews_metric(y_true, y_pred)
        elif xm == 'hlm':
            xmet =hamming_metric(y_true, y_pred)
        else:
            xmet = 0

        res_dict[xm] = xmet

        xsum = xsum + xmet
        xcont = xcont +1

    if 'sum' in aggregates:
        res_dict['sum'] = xsum
    if 'avg' in aggregates and xcont > 0:
        res_dict['avg'] = xsum/xcont
    # Ask for arguments for each metric

    return res_dict
#------------------------------------------------------------------------------------------------------------------

def process_target(xdf_data, target_type, encoder):
    '''
        1- Multiclass  target = (1...n, text1...textn)
        2- Multilabel target = ( list(Text1, Text2, Text3 ) for each observation, separated by commas )
        3- Multiclass or Multilabel ( binary  ( 0,1 ) )
    :return:
    '''


    if target_type == 1:
        # takes the classes and then
        class_names:List[str] = list(xdf_data['label'].unique())
        encoder.fit(class_names)
        final_target:np.ndarray = encoder.transform(xdf_data['label'])
        xdf_data['target_class'] = final_target


    if target_type == 2:
        target = np.array(xdf_data['label'].apply( lambda x : x.split(",")))
        final_target = encoder.fit_transform(target)
        xfinal = []
        if len(final_target) ==0:
            xerror = 'Could not process Multilabel'
            class_names = []
        else:
            class_names = encoder.classes_
            for i in range(len(final_target)): 
                joined_string = ",".join( str(e) for e in final_target[i])
                xfinal.append(joined_string)
            xdf_data['target_class'] = xfinal

    if target_type == 3:
        target = np.array(xdf_data['label'].apply(lambda x: x.split(",")))
        final_target = encoder.fit_transform(target)
        xfinal = []
        if len(final_target) == 0:
            xerror = 'Could not process Multilabel'
            class_names = []
        else:
            class_names = encoder.classes_

    ## We add the column to the main dataset
    return class_names
#------------------------------------------------------------------------------------------------------------------

def data_augmentation(img:torch.Tensor, n_class:int):
    """function to increase sample size for minority classes"""
    count:List[int] = [0] * n_class
    pass
    

#------------------------------------------------------------------------------------------------------------------
def main() -> None:
    # add pip install https://github.com/ufoym/imbalanced-dataset-sampler/archive/master.zip
    OR_PATH:str = os.getcwd()
    DATA_DIR:str = os.getcwd() + os.path.sep + 'data' + os.path.sep + "images"\
        + os.path.sep
    
    util:Utility = Utility()
    device:str = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    for file_name in os.listdir(OR_PATH +os.path.sep + "references"):
        if file_name == "train.csv":
            train_path:str = OR_PATH+ os.path.sep+ "references" + os.path.sep\
                + file_name
        else:
            test_path:str = OR_PATH+ os.path.sep+ "references" + os.path.sep\
                + file_name

    # Reading and filtering Excel file
    xdf_data_train:pd.DataFrame = pd.read_csv(train_path)
    xdf_dset_test:pd.DataFrame = pd.read_csv(test_path)

    

    ## Process Classes
    ## Input and output

    ## Processing Train dataset
    ## Target_type = 1  Multiclass   Target_type = 2 MultiLabel
    le:LabelEncoder = LabelEncoder()
    class_names:Tuple[int] = process_target(
        xdf_data_train, 
        target_type = 1, 
        encoder = le
    )

    # data_augmentation()
    
    # separate train and valid excel file
    _train_test_tuple:Tuple[pd.DataFrame, pd.DataFrame] = train_test_split(
        xdf_data_train,
        train_size = 0.8,
        random_state = 40,
        stratify = xdf_data_train["label"]
    )
    xdf_dset_train:pd.DataFrame = _train_test_tuple[0]
    xdf_dset_valid:pd.DataFrame = _train_test_tuple[1]

    ## read_data creates the dataloaders, take target_type = 2

    n_class:int = len(class_names)
    train_ds, valid_ds = read_data(
        util.BATCH_SIZE,
        util.IMAGE_SIZE_1, 
        util.IMAGE_SIZE_2, 
        DATA_DIR, 
        xdf_dset_train,
        xdf_dset_valid, 
        xdf_dset_test, 
        n_class, 
        target_type = 1
    )


    list_of_metrics = ["acc", 'f1_macro', 'coh']
    list_of_agg = ['sum']

    train_and_test(train_ds, valid_ds, list_of_metrics, list_of_agg, util.n_epoch, util.LR, util.model_name, device, xdf_dset_valid, n_class, OR_PATH, SAVE_MODEL = True, save_on='sum')


if __name__ == '__main__':
    main()
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
from utility_cnn_viz import Utility
from sklearn.model_selection import train_test_split
from torchsummary import summary
import matplotlib.pyplot as plt
from PIL import Image

# ------------------------------------------------------------------------------------------------------------------


# ------------------------------------------------------------------------------------------------------------------


# ------------------------------------------------------------------------------------------------------------------

class CNN(nn.Module):

    def __init__(self, OUTPUTS_a: int):
        super().__init__()
        self.conv1: nn.Conv2d = nn.Conv2d(3, 16, (7, 7), stride=2)
        self.convnorm1: nn.BatchNorm2d = nn.BatchNorm2d(16)
        self.pool1: nn.MaxPool2d = nn.MaxPool2d((3, 3), stride=2)
        # self.pad1: nn.ZeroPad2d = nn.ZeroPad2d(2)

        self.conv2: nn.Conv2d = nn.Conv2d(16, 64, (5, 5), stride=2)
        self.convnorm2: nn.BatchNorm2d = nn.BatchNorm2d(64)
        self.pool2: nn.MaxPool2d = nn.MaxPool2d((3, 3), stride=2)

        self.conv3: nn.Conv2d = nn.Conv2d(64, 128, (3, 3), stride=1)
        self.convnorm3: nn.BatchNorm2d = nn.BatchNorm2d(128)
        # self.pool3: nn.MaxPool2d = nn.MaxPool2d((3, 3), stride=2)

        self.conv4: nn.Conv2d = nn.Conv2d(128, 256, (3, 3), stride=1)

        # self.conv5: nn.Conv2d = nn.Conv2d(384, 256, (2, 2),stride=1)
        # self.pool5: nn.MaxPool2d = nn.MaxPool2d((3, 3), stride=2)
        self.global_avg_pool: nn.AdaptiveAvgPool2d = nn.AdaptiveAvgPool2d((1, 1))

        self.linear1: nn.Linear = nn.Linear(256, 4096)

        self.linear2: nn.Linear = nn.Linear(4096, OUTPUTS_a)
        self.act: nn.ReLU = nn.ReLU()
        # self.Softmax: nn.Softmax = nn.Softmax(dim=1)
        self.norm: nn.BatchNorm1d = nn.BatchNorm1d(OUTPUTS_a)

    def forward(self, x: torch.TensorType):
        x = self.pool1(self.convnorm1(self.act(self.conv1(x))))
        x = self.pool2(self.convnorm2(self.act(self.conv2(x))))
        x = self.act(self.conv4(self.convnorm3(self.act(self.conv3(x)))))
        # x = self.act(self.conv5(x))
        x = self.act(self.linear1(self.global_avg_pool(x).view(-1, 256)))
        x = self.norm(self.linear2(x))
        return x


class convnet(nn.Module):

    def __init__(self, OUTPUTS_a: int):
        super().__init__()
        self.conv1: nn.Conv2d = nn.Conv2d(3, 16, (7, 7))
        self.convnorm1: nn.BatchNorm2d = nn.BatchNorm2d(16)
        self.pad1: nn.ZeroPad2d = nn.ZeroPad2d(2)

        self.conv2: nn.Conv2d = nn.Conv2d(16, 20, (3, 3))
        self.convnorm2: nn.BatchNorm2d = nn.BatchNorm2d(20)
        self.pool2: nn.MaxPool2d = nn.MaxPool2d((2, 2))

        self.conv3: nn.Conv2d = nn.Conv2d(20, 20, (3, 3))
        self.convnorm3: nn.BatchNorm2d = nn.BatchNorm2d(64)

        self.conv4: nn.Conv2d = nn.Conv2d(64, 128, (3, 3))
        self.global_avg_pool: nn.AdaptiveAvgPool2d = nn.AdaptiveAvgPool2d((1, 1))

        self.linear: nn.Linear = nn.Linear(128, OUTPUTS_a)
        self.act: nn.ReLU = nn.ReLU
        self.Softmax: nn.Softmax = nn.Softmax(dim=1)
        self.norm: nn.BatchNorm1d = nn.BatchNorm1d(OUTPUTS_a)

    def forward(self, x: torch.TensorType):
        x = self.pad1(self.convnorm1(self.act(self.conv1(x))))
        x = self.pool2(self.convnorm2(self.act(self.conv2(x))))
        x = self.act(self.conv4(self.convnorm3(self.act(self.conv3(x)))))
        x = self.norm(self.linear(self.global_avg_pool(x).view(-1, 128)))
        return self.Softmax(x)


# ------------------------------------------------------------------------------------------------------------------


class My_resnet50(nn.Module):
    def __init__(self, OUTPUTS_a: int) -> None:
        super().__init__()

        model = models.resnet50(pretrained=True)
        model.fc = nn.Sequential(
            nn.Linear(model.fc.in_features, OUTPUTS_a),
            nn.Softmax(dim=1)
        )
        self.resnet50 = model

    def forward(self, x: torch.TensorType):
        x = self.resnet50(x)
        # x = self.new(x)
        return x


# ------------------------------------------------------------------------------------------------------------------


class Custom_Dataset(data.Dataset):
    '''
    From : https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel
    '''

    def __init__(
            self,
            list_IDs: List[Any],
            type_data: str,
            target_type: int,
            IMAGE_SIZE_1: int,
            IMAGE_SIZE_2: int,
            DATA_DIR: str,
            xdf_dset_train: pd.DataFrame,
            xdf_dset_valid: pd.DataFrame,
            xdf_dset_test: pd.DataFrame,
            OUTPUTS_a: int) -> None:

        super().__init__()

        # Initialization'
        self.list_IDs: List[Any] = list_IDs
        self.type_data: str = type_data
        self.target_type: int = target_type
        self.IMAGE_SIZE_1: int = IMAGE_SIZE_1
        self.IMAGE_SIZE_2: int = IMAGE_SIZE_2
        self.DATA_DIR: str = DATA_DIR
        self.xdf_dset_train: pd.DataFrame = xdf_dset_train
        self.xdf_dset_valid: pd.DataFrame = xdf_dset_valid
        self.xdf_dset_test: pd.DataFrame = xdf_dset_test
        self.OUTPUTS_a: int = OUTPUTS_a

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

    def __getitem__(self, index: int):
        # Generates one sample of data'
        # Select sample
        ID: int = self.list_IDs[index]
        # Load data and get label

        if self.type_data == 'train':
            file: str = self.DATA_DIR \
                        + self.xdf_dset_train.loc[ID, "image"].replace("images/", "")
            y: Union[str, int] = self.xdf_dset_train.loc[ID, "target_class"]

        elif self.type_data == "valid":
            file = self.DATA_DIR \
                   + self.xdf_dset_valid.loc[ID, "image"].replace("images/", "")
            y = self.xdf_dset_valid.loc[ID, "target_class"]
        else:
            file = self.DATA_DIR \
                   + self.xdf_dset_test.loc[ID, "image"].replace("images/", "")

        if self.target_type == 2:
            y_lst: List[str] = y.split(",")
            labels_ml: List[int] = [int(e) for e in y_lst]
            y_tensor: torch.Tensor = torch.FloatTensor(labels_ml)
        # else:
        #     labels_mc: np.ndarray = np.zeros(self.OUTPUTS_a)
        #     for _idx, _label in enumerate(range(self.OUTPUTS_a)):
        #         if _label == y:
        #             labels_mc[_idx] = 1
        #     y_tensor = torch.FloatTensor(labels_mc)

        img: np.ndarray = cv2.imread(file)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img: torch.Tensor = self.transform(img)

        img = torch.FloatTensor(img)

        # Augmentation only for train

        return img, y


# ------------------------------------------------------------------------------------------------------------------


def read_data(
        BATCH_SIZE: int,
        IMAGE_SIZE_1: int,
        IMAGE_SIZE_2: int,
        DATA_DIR: str,
        xdf_dset_train: pd.DataFrame,
        xdf_dset_valid: pd.DataFrame,
        xdf_dset_test: pd.DataFrame,
        OUTPUTS_a: int,
        target_type: int):
    # ---------------------- Parameters for the data loader --------------------------------

    list_of_ids_train: List[pd.RangeIndex] = list(xdf_dset_train.index)
    list_of_ids_valid: List[pd.RangeIndex] = list(xdf_dset_valid.index)
    list_of_ids_test: List[pd.RangeIndex] = list(xdf_dset_test.index)

    # Data Loaders

    params: Dict[str, Union[int, bool]] = {
        'batch_size': BATCH_SIZE,
        'shuffle': True
    }

    training_set: Custom_Dataset = Custom_Dataset(
        list_of_ids_train,
        'train',
        target_type,
        IMAGE_SIZE_1,
        IMAGE_SIZE_2,
        DATA_DIR,
        xdf_dset_train,
        xdf_dset_valid,
        xdf_dset_test,
        OUTPUTS_a
    )

    training_generator: data.DataLoader = data.DataLoader(training_set, **params)

    params: Dict[str, Union[int, bool]] = {
        'batch_size': BATCH_SIZE,
        'shuffle': False
    }

    validation_set: Custom_Dataset = Custom_Dataset(
        list_of_ids_valid,
        'valid',
        target_type,
        IMAGE_SIZE_1,
        IMAGE_SIZE_2,
        DATA_DIR,
        xdf_dset_train,
        xdf_dset_valid,
        xdf_dset_test,
        OUTPUTS_a
    )

    validation_generator: data.DataLoader = data.DataLoader(validation_set, **params)

    params: Dict[str, Union[int, bool]] = {
        'batch_size': BATCH_SIZE,
        'shuffle': False
    }

    test_set: Custom_Dataset = Custom_Dataset(
        list_of_ids_test,
        'test',
        target_type,
        IMAGE_SIZE_1,
        IMAGE_SIZE_2,
        DATA_DIR,
        xdf_dset_train,
        xdf_dset_valid,
        xdf_dset_test,
        OUTPUTS_a
    )

    test_generator: data.DataLoader = data.DataLoader(test_set, **params)

    # Make the channel as a list to make it variable

    return training_generator, validation_generator, test_generator


# ------------------------------------------------------------------------------------------------------------------


def save_model(model, NICKNAME: str) -> None:
    '''
      Print Model Summary
    '''

    print(model, file=open(PATH + os.path.sep + 'models/summary_{}.txt'.format(NICKNAME), "w"))


# ------------------------------------------------------------------------------------------------------------------

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        model = model
        for param in model.parameters():
            param.requires_grad = False


def model_definition(
        LR: float,
        NICKNAME: str,
        device: str,
        OUTPUTS_a: int,
        pretrain: bool):
    '''
        Define a Keras sequential model
        Compile the model
    '''

    if pretrain:
        # model: My_resnet50 = My_resnet50(OUTPUTS_a)

        #'''

        model = models.vgg16(pretrained=True)
        model.classifier = nn.Sequential(
            nn.Linear(25088, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, OUTPUTS_a),
            #nn.Softmax(dim=1)
        )

        for name, param in model.named_parameters():
            if 'classifier' not in name:
                param.requires_grad = False
        #'''
        # resnet34
        # model = models.resnet34(pretrained=pretrain)
        #model = models.resnet18(pretrained=True)

        # set_parameter_requires_grad(model, False)

        #model.fc = nn.Sequential(nn.Linear(model.fc.in_features, OUTPUTS_a))
        # '''
    else:
        model: CNN = CNN(OUTPUTS_a)

    print(summary(model, (3, 224, 224)))

    model = model.to(device)

    # optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9)
    weight_decay = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=weight_decay)

    # criterion: nn.CrossEntropyLoss = nn.BCELoss() #which one is used
    # criterion = nn.BCELoss()
    criterion: nn.CrossEntropyLoss = nn.CrossEntropyLoss()

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True
    )

    save_model(model, NICKNAME)

    return model, optimizer, criterion, scheduler


# ------------------------------------------------------------------------------------------------------------------

def train_and_test(
        train_ds,
        valid_ds,
        list_of_metrics: List[str],
        list_of_agg: List[str],
        n_epoch: int,
        LR: float,
        NICKNAME: str,
        device: str,
        xdf_dset_valid: pd.DataFrame,
        OUTPUTS_a: int,
        OR_PATH: str,
        SAVE_MODEL: bool,
        save_on: str,
        pretrain: bool = False):
    model, optimizer, criterion, scheduler = model_definition(
        LR,
        NICKNAME,
        device,
        OUTPUTS_a,
        pretrain
    )

    cont: int = 0
    train_loss_item: List = []
    valid_loss_item: List = []
    total_valid_loss: List = []

    model.phase = 0

    met_valid_best: float = 0
    for epoch in range(n_epoch):
        train_loss, steps_train = 0, 0

        model.train()
        pred_labels_per_hist: List[float] = []
        pred_logits: List[float] = []
        train_hist: List[float] = []
        valid_hist: List[float] = []

        with tqdm(total=len(train_ds), desc="Epoch {}".format(epoch)) as pbar:

            for xdata, xtarget in train_ds:
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

                pred_logits = pred_logits + list(np.argmax(pred_labels_per, axis=1))

        # Metric Evaluation
        train_metrics = metrics_func(list_of_metrics, list_of_agg, train_hist, pred_logits)

        avg_train_loss = train_loss / steps_train

        ## Finish with Training

        ## Testing the model

        model.eval()

        pred_labels_per_hist_valid: List[float] = []

        pred_logits_valid: List[int] = []

        valid_loss, steps_valid = 0, 0
        met_valid = 0

        with torch.no_grad():

            with tqdm(total=len(valid_ds), desc="Epoch {}".format(epoch)) as pbar:
                for xdata, xtarget in valid_ds:
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

                    pred_logits_valid = pred_logits_valid + list(np.argmax(pred_labels_per, axis=1))

        valid_metrics = metrics_func(list_of_metrics, list_of_agg, valid_hist, pred_logits_valid)

        avg_valid_loss = valid_loss / steps_valid

        scheduler.step(avg_valid_loss)
        total_valid_loss.append(avg_valid_loss)

        xstrres = "Epoch {}: ".format(epoch)
        for met, dat in train_metrics.items():
            xstrres = xstrres + ' Train ' + met + ' {:.5f}'.format(dat)

        xstrres = xstrres + " - "
        for met, dat in valid_metrics.items():
            xstrres = xstrres + ' Validation ' + met + ' {:.5f}'.format(dat)
            if met == save_on:
                met_valid = dat

        print(xstrres)  # Print metrics

        if met_valid > met_valid_best and SAVE_MODEL:
            torch.save(model.state_dict(), PATH + os.path.sep + "models/model_{}.pt".format(NICKNAME))
            # torch.save(model.state_dict(), OR_PATH + os.path.sep
            #            + "models" + os.path.sep + "model.pt")
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


# ------------------------------------------------------------------------------------------------------------------

def test_model(test_ds, list_of_metrics, list_of_agg,
               LR,
               NICKNAME: str,
               device: str,
               OUTPUTS_a,
               pretrain: bool = False):
    model, optimizer, criterion, scheduler = model_definition(LR,NICKNAME,device,OUTPUTS_a,pretrain)
    #model: CNN = CNN(OUTPUTS_a)

    print(summary(model, (3, 224, 224)))

    model = model.to(device)
    model.load_state_dict(torch.load(PATH + os.path.sep + "models/model_{}.pt".format(NICKNAME), map_location=device))

    criterion: nn.CrossEntropyLoss = nn.CrossEntropyLoss()

    # pred_logits, real_labels = np.zeros((1, OUTPUTS_a)), np.zeros((1, OUTPUTS_a))

    model.eval()

    cont: int = 0
    model.phase = 0
    pred_labels_per_hist: List[float] = []
    pred_logits: List[float] = []
    test_hist: List[float] = []

    pred_labels_per_hist_valid: List[float] = []

    pred_logits_test: List[int] = []

    test_loss, steps_test = 0, 0
    met_test = 0

    with torch.no_grad():
        with tqdm(total=len(test_ds)) as pbar:
            for xdata, xtarget in test_ds:
                xdata, xtarget = xdata.to(device), xtarget.to(device)
                # optimizer.zero_grad()
                output = model(xdata)

                loss = criterion(output, xtarget)
                test_loss += loss.item()
                cont += 1

                steps_test += 1

                # test_loss_item.append([epoch, loss.item()])

                pred_labels_per = output.detach().to(torch.device('cpu')).numpy()

                # pred_labels_per_hist = pred_labels_per_hist + list(pred_labels_per)

                test_hist_per = xtarget.cpu().numpy()

                test_hist = test_hist + list(test_hist_per)

                pbar.update(1)
                pbar.set_postfix_str("Validation Loss: {:.5f}".format(test_loss / steps_test))

                pred_logits_test = pred_logits_test + list(np.argmax(pred_labels_per, axis=1))

    test_metrics = metrics_func(list_of_metrics, list_of_agg, test_hist, pred_logits_test)

    avg_valid_loss = test_loss / steps_test

    # total_valid_loss.append(avg_valid_loss)

    xstrres = ""
    for met, dat in test_metrics.items():
        xstrres = xstrres + ' Test ' + met + ' {:.5f}'.format(dat)

    print(xstrres)  # Print metrics

    # xres = [f.numpy() for f in pred_logits_test]
    xdf_dset_valid['results'] = pred_logits_test
    xdf_dset_valid.to_excel(PATH + os.path.sep + "models/results_{}.xlsx".format(NICKNAME), index=False)
    print("The model has been saved!")


def plot_model(NICKNAME: str,
              device: str,
    plot_filter = False, plot_feature = True):

    model: CNN = CNN(OUTPUTS_a)

    model = model.to('cpu')
    model.load_state_dict(torch.load(PATH + os.path.sep + "models/model_{}.pt".format(NICKNAME), map_location='cpu'))

    model_weights = []  # we will save the conv layer weights in this list
    conv_layers = []  # we will save the 49 conv layers in this list
    # get all the model children as list
    model_children = list(model.children())

    # counter to keep count of the conv layers
    counter = 0
    # append all the conv layers and their respective weights to the list
    for i in range(len(model_children)):
        if type(model_children[i]) == nn.Conv2d:
            counter += 1
            model_weights.append(model_children[i].weight)
            conv_layers.append(model_children[i])
        elif type(model_children[i]) == nn.Sequential:
            for j in range(len(model_children[i])):
                for child in model_children[i][j].children():
                    if type(child) == nn.Conv2d:
                        counter += 1
                        model_weights.append(child.weight)
                        conv_layers.append(child)
    # print(f"Total convolutional layers: {counter}")

    # take a look at the conv layers and the respective weights
    for weight, conv in zip(model_weights, conv_layers):
        # print(f"WEIGHT: {weight} \nSHAPE: {weight.shape}")
        print(f"CONV: {conv} ====> SHAPE: {weight.shape}")

    if plot_filter:
        #'''
        for j, model_weight, layer_channel in zip(range(4), model_weights, [4, 8, 8, 8]):
            # visualize the first conv layer filters
            print(j)
            plt.figure(figsize=(20, 17))
            print(len(model_weight))

            for i, filter in enumerate(model_weight[:layer_channel**2]):
                filter = np.asarray(filter.detach().numpy())
                plt.subplot(layer_channel, layer_channel, i + 1)
                #plt.subplot(layer_channel, layer_channel, i + 1)  # (8, 8) because in conv0 we have 7x7 filters and total of 64 (see printed shapes)
                if j !=0:
                    plt.imshow(filter[0, :, :], cmap='gray')
                    plt.axis('off')
                    #plt.savefig(PATH + os.path.sep + f'data/visualization/filter_map/filter_{j}_{NICKNAME}.png')
                if j ==0:
                    #img = Image.fromarray(filter[:3, :, :], 'RGB')
                    #plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                    #print(filter.shape())
                    #plt.imshow(cv2.cvtColor(filter, cv2.COLOR_BGR2RGB))
                    filter_rgb= np.einsum('kij->ijk', filter)
                    #print(filter_rgb)
                    filter_rgb = (filter_rgb - np.min(filter_rgb)) / np.ptp(filter_rgb) * np.array([[[255]*3]*7]*7)
                    filter_rgb = filter_rgb.astype(int)
                    #print(filter_rgb)
                    '''
                    filter_rgb = np.array(np.zeros((7,7,3)))
                    for ii in range(7):
                        for jj in range(7):
                            filter_rgb[ii,jj,0] = filter[0,ii,jj]
                            filter_rgb[ii, jj, 1] = filter[1, ii, jj]
                            filter_rgb[ii, jj, 2] = filter[2, ii, jj]
                    '''
                    plt.imshow(filter_rgb)
                    plt.axis('off')
                    #numpy.asarray(
            if j==0:
                plt.savefig(PATH + os.path.sep + f'data/visualization/filter_map/filter_{j}_RGB_{NICKNAME}.png')
            if j!=0:
                plt.savefig(PATH + os.path.sep + f'data/visualization/filter_map/filter_{j}_{NICKNAME}.png')
            #if j ==0:
            plt.show()
        #'''
    if plot_feature:
        # read and visualize an image
        img = cv2.imread(DATA_DIR + '0.jpg')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img)
        plt.show()
        # define the transforms
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((util.IMAGE_SIZE_1, util.IMAGE_SIZE_2)),
            transforms.ToTensor(),
        ])
        img = np.array(img)
        # apply the transforms
        img = transform(img)
        print(img.size())
        # unsqueeze to add a batch dimension
        img = img.unsqueeze(0)
        print(img.size())

        #model_feature_plot = [m for m in model.modules()][:4]
        model_feature_plot = model_children[:5]
        #model_feature_plot = conv_layers

        print(model_feature_plot)

        # pass the image through all the layers
        results = [model_feature_plot[0](img)]
        for i in range(1, len(model_feature_plot)):
            # pass the result from the last layer to the next layer
            print(model_feature_plot[i])
            results.append(model_feature_plot[i](results[-1]))
        # make a copy of the `results`
        outputs = results

        # visualize 64 features from each layer
        # (although there are more feature maps in the upper layers)

        for num_layer in range(len(outputs)):
            plt.figure(figsize=(30, 30))
            layer_viz = outputs[num_layer][0, :, :, :]
            layer_viz = layer_viz.data
            print(layer_viz.size())
            for i, filter in enumerate(layer_viz):
                if i == 64:  # we will visualize only 8x8 blocks from each layer
                    break
                if len(layer_viz) == 16:
                    plt.subplot(4, 4, i + 1)
                    plt.imshow(filter, cmap='gray')
                    plt.axis("off")
                else:
                    plt.subplot(8, 8, i + 1)
                    plt.imshow(filter, cmap='gray')
                    plt.axis("off")
            print(f"Saving layer {num_layer} feature maps...")
            plt.savefig(PATH + os.path.sep + f'data/visualization/feature_map/feature_{num_layer}_{NICKNAME}.png')
            # '''


# ------------------------------------------------------------------------------------------------------------------


# ------------------------------------------------------------------------------------------------------------------


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

    xcont = 0
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
            xmet = accuracy_metric(y_true, y_pred)
        elif xm == 'mat':
            # Matthews
            xmet = matthews_metric(y_true, y_pred)
        elif xm == 'hlm':
            xmet = hamming_metric(y_true, y_pred)
        else:
            xmet = 0

        res_dict[xm] = xmet

        xsum = xsum + xmet
        xcont = xcont + 1

    if 'sum' in aggregates:
        res_dict['sum'] = xsum
    if 'avg' in aggregates and xcont > 0:
        res_dict['avg'] = xsum / xcont
    # Ask for arguments for each metric

    return res_dict


# ------------------------------------------------------------------------------------------------------------------

def process_target(xdf_data, target_type, encoder):
    '''
        1- Multiclass  target = (1...n, text1...textn)
        2- Multilabel target = ( list(Text1, Text2, Text3 ) for each observation, separated by commas )
        3- Multiclass or Multilabel ( binary  ( 0,1 ) )
    :return:
    '''

    if target_type == 1:
        # takes the classes and then
        class_names: List[str] = list(xdf_data['label'].unique())
        encoder.fit(class_names)
        final_target: np.ndarray = encoder.transform(xdf_data['label'])
        xdf_data['target_class'] = final_target

    if target_type == 2:
        target = np.array(xdf_data['label'].apply(lambda x: x.split(",")))
        final_target = encoder.fit_transform(target)
        xfinal = []
        if len(final_target) == 0:
            xerror = 'Could not process Multilabel'
            class_names = []
        else:
            class_names = encoder.classes_
            for i in range(len(final_target)):
                joined_string = ",".join(str(e) for e in final_target[i])
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


# ------------------------------------------------------------------------------------------------------------------
# def main() -> None:
if __name__ == '__main__':
    OR_PATH: str = os.getcwd()
    os.chdir("..")  # Change to the parent directory
    PATH = os.getcwd()
    DATA_DIR: str = os.getcwd() + os.path.sep + 'data' + os.path.sep + "images" + os.path.sep

    util: Utility = Utility()
    device: str = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # for file_name in os.listdir(PATH + os.path.sep + "references"):
    train_path: str = PATH + os.path.sep + "references" + os.path.sep + "train.csv"
    test_path: str = PATH + os.path.sep + "references" + os.path.sep + "test.csv"

    # Reading and filtering Excel file
    xdf_data_train: pd.DataFrame = pd.read_csv(train_path)
    xdf_dset_test: pd.DataFrame = pd.read_csv(test_path)

    ## Process Classes
    ## Input and output

    ## Processing Train dataset
    ## Target_type = 1  Multiclass   Target_type = 2 MultiLabel
    le: LabelEncoder = LabelEncoder()
    class_names: Tuple[int] = process_target(
        xdf_data_train,
        target_type=1,
        encoder=le
    )

    # separate train and valid excel file
    xdf_dset_train, xdf_dset_valid = train_test_split(
        xdf_data_train,
        train_size=0.8,
        random_state=40,
        stratify=xdf_data_train["label"]
    )

    ## read_data creates the dataloaders, take target_type = 2

    OUTPUTS_a: int = len(class_names)
    train_ds, valid_ds, test_ds = read_data(
        util.BATCH_SIZE,
        util.IMAGE_SIZE_1,
        util.IMAGE_SIZE_2,
        DATA_DIR,
        xdf_dset_train,
        xdf_dset_valid,
        xdf_dset_test,
        OUTPUTS_a,
        target_type=1
    )

    list_of_metrics = ['acc', 'f1_macro', 'coh']
    list_of_agg = ['sum']

    test_model(valid_ds, list_of_metrics, list_of_agg, util.LR, util.NICKNAME, device, OUTPUTS_a,pretrain=False)

    plot_model(util.NICKNAME, device, plot_filter=False, plot_feature=True)

    #train_and_test(train_ds, valid_ds, list_of_metrics, list_of_agg, util.n_epoch, util.LR, util.NICKNAME, device,
      #xdf_dset_valid, OUTPUTS_a, OR_PATH, SAVE_MODEL=True, save_on='sum', pretrain=True)
    print('finished')

# if __name__ == '__main__':
# main()
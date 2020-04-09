#!/usr/bin/env python

##  object_detection_and_localization.py

"""
This script shows how you can use the functionality provided by the inner class
DetectAndLocalize of the DLStudio module for experimenting with object detection and
localization.

Detecting and localizing objects in images is a more difficult problem than just
classifying the objects.  The former requires that your CNN make two different types
of inferences simultaneously, one for classification and the other for localization.
For the localization part, the CNN must carry out what is known as regression. What
that means is that the CNN must output the numerical values for the bounding box that
encloses the object that was detected.  Generating these two types of inferences
requires two different loss functions, one for classification and the other for
regression.

Training a CNN to solve the detection and localization problem requires a dataset
that, in addition to the class labels for the objects, also provides bounding-box
annotations for the objects in the images.  As you see in the code below, this
script uses the PurdueShapes5 dataset for that purpose.
"""

import torch
import torchvision
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.transforms as tvt
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.datasets import ImageFolder
from torchsummary import summary
from pathlib import Path
import gzip
import pickle
import os, sys
import random
import pymsgbox
import copy
from PIL import ImageFilter
import numbers
import re
import math
import random
import torch.optim as optim

seed = 0
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmarks = False
os.environ['PYTHONHASHSEED'] = str(seed)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dtype = torch.float


def gaussian_filter(shape, mu_x, mu_y, sigma_x, sigma_y):
    m = n = shape // 2
    h = torch.zeros(3, 1, shape, shape)
    for index in range(3):
        for x in [i for i in range(-m, m + 1, 1)]:
            for y in [j for j in range(-n, n + 1, 1)]:
                h[index][0][x + 1][y + 1] = torch.exp(
                    -(((x - mu_x) ** 2) / (2. * sigma_x * sigma_x) + ((y - mu_y) ** 2) / (2. * sigma_y * sigma_y)))
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    h = h.float()
    return h


def apply_filter(img, filter):
    out = F.conv2d(img, filter, padding=1, groups=3)
    return out


class DLStudio(object):

    def __init__(self, *args, **kwargs):
        if args:
            raise ValueError(
                '''DLStudio constructor can only be called with keyword arguments for 
                      the following keywords: epochs, learning_rate, batch_size, momentum,
                      convo_layers_config, image_size, dataroot, path_saved_model, classes, 
                      image_size, convo_layers_config, fc_layers_config, debug_train, use_gpu, and 
                      debug_test''')
        learning_rate = epochs = batch_size = convo_layers_config = momentum = None
        image_size = fc_layers_config = dataroot = path_saved_model = classes = use_gpu = None
        debug_train = debug_test = None
        if 'dataroot' in kwargs:   dataroot = kwargs.pop('dataroot')
        if 'learning_rate' in kwargs:   learning_rate = kwargs.pop('learning_rate')
        if 'momentum' in kwargs:   momentum = kwargs.pop('momentum')
        if 'epochs' in kwargs:   epochs = kwargs.pop('epochs')
        if 'batch_size' in kwargs:   batch_size = kwargs.pop('batch_size')
        if 'convo_layers_config' in kwargs:   convo_layers_config = kwargs.pop('convo_layers_config')
        if 'image_size' in kwargs:   image_size = kwargs.pop('image_size')
        if 'fc_layers_config' in kwargs:   fc_layers_config = kwargs.pop('fc_layers_config')
        if 'path_saved_model' in kwargs:   path_saved_model = kwargs.pop('path_saved_model')
        if 'classes' in kwargs:   classes = kwargs.pop('classes')
        if 'use_gpu' in kwargs:   use_gpu = kwargs.pop('use_gpu')
        if 'debug_train' in kwargs:   debug_train = kwargs.pop('debug_train')
        if 'debug_test' in kwargs:   debug_test = kwargs.pop('debug_test')
        if len(kwargs) != 0: raise ValueError('''You have provided unrecognizable keyword args''')
        if dataroot:
            self.dataroot = dataroot
        if convo_layers_config:
            self.convo_layers_config = convo_layers_config
        if image_size:
            self.image_size = image_size
        if fc_layers_config:
            self.fc_layers_config = fc_layers_config
            if fc_layers_config[0] is not -1:
                raise Exception("""\n\n\nYour 'fc_layers_config' construction option is not correct. """
                                """The first element of the list of nodes in the fc layer must be -1 """
                                """because the input to fc will be set automatically to the size of """
                                """the final activation volume of the convolutional part of the network""")
        if path_saved_model:
            self.path_saved_model = path_saved_model
        if classes:
            self.class_labels = classes
        if learning_rate:
            self.learning_rate = learning_rate
        else:
            self.learning_rate = 1e-6
        if momentum:
            self.momentum = momentum
        if epochs:
            self.epochs = epochs
        if batch_size:
            self.batch_size = batch_size
        if use_gpu is not None:
            self.use_gpu = use_gpu
            if use_gpu is True:
                if torch.cuda.is_available():
                    self.device = torch.device("cuda:0")
                else:
                    raise Exception("You requested GPU support, but there's no GPU on this machine")
            else:
                self.device = torch.device("cpu")
        if debug_train:
            self.debug_train = debug_train
        else:
            self.debug_train = 0
        if debug_test:
            self.debug_test = debug_test
        else:
            self.debug_test = 0
        self.debug_config = 0


#        self.device = torch.device("cuda:0" if torch.cuda.is_available() and self.use_gpu is False else "cpu")

class DetectAndLocalize(nn.Module):
    """
        The purpose of this inner class is to focus on object detection in images --- as
        opposed to image classification.  Most people would say that object detection
        is a more challenging problem than image classification because, in general,
        the former also requires localization.  The simplest interpretation of what
        is meant by localization is that the code that carries out object detection
        must also output a bounding-box rectangle for the object that was detected.

        You will find in this inner class some examples of LOADnet classes meant
        for solving the object detection and localization problem.  The acronym
        "LOAD" in "LOADnet" stands for

                    "LOcalization And Detection"

        The different network examples included here are LOADnet1, LOADnet2, and
        LOADnet3.  For now, only pay attention to LOADnet2 since that's the class I
        have worked with the most for the 1.0.7 distribution.
        """

    def __init__(self, dl_studio, dataserver_train=None, dataserver_test=None, dataset_file_train=None,
                 dataset_file_test=None):
        super(DetectAndLocalize, self).__init__()
        self.dl_studio = dl_studio
        self.dataserver_train = dataserver_train
        self.dataserver_test = dataserver_test

    class PurdueShapes5Dataset(torch.utils.data.Dataset):
        def __init__(self, dl_studio, train_or_test, dataset_file, transform=None):
            super(DetectAndLocalize.PurdueShapes5Dataset, self).__init__()
            if train_or_test == 'train' and dataset_file == "PurdueShapes5-10000-train.gz":
                if os.path.exists("torch-saved-PurdueShapes5-10000-dataset.pt") and \
                        os.path.exists("torch-saved-PurdueShapes5-label-map.pt"):
                    print("\nLoading training data from the torch-saved archive")
                    self.dataset = torch.load("torch-saved-PurdueShapes5-10000-dataset.pt")
                    self.label_map = torch.load("torch-saved-PurdueShapes5-label-map.pt")
                    # reverse the key-value pairs in the label dictionary:
                    self.class_labels = dict(map(reversed, self.label_map.items()))
                    self.transform = transform
                else:
                    print("""\n\n\nLooks like this is the first time you will be loading in\n"""
                          """the dataset for this script. First time loading could take\n"""
                          """a minute or so.  Any subsequent attempts will only take\n"""
                          """a few seconds.\n\n\n""")
                    root_dir = dl_studio.dataroot
                    f = gzip.open(root_dir + dataset_file, 'rb')
                    dataset = f.read()
                    if sys.version_info[0] == 3:
                        self.dataset, self.label_map = pickle.loads(dataset, encoding='latin1')
                    else:
                        self.dataset, self.label_map = pickle.loads(dataset)
                    torch.save(self.dataset, "torch-saved-PurdueShapes5-10000-dataset.pt")
                    torch.save(self.label_map, "torch-saved-PurdueShapes5-label-map.pt")
                    # reverse the key-value pairs in the label dictionary:
                    self.class_labels = dict(map(reversed, self.label_map.items()))
                    self.transform = transform
            elif train_or_test == 'train' and dataset_file == "PurdueShapes5-10000-train-noise-20.gz":
                if os.path.exists("torch-saved-PurdueShapes5-10000-dataset-noise-20.pt") and \
                        os.path.exists("torch-saved-PurdueShapes5-label-map.pt"):
                    print("\nLoading training data from the torch-saved archive")
                    self.dataset = torch.load("torch-saved-PurdueShapes5-10000-dataset-noise-20.pt")
                    self.label_map = torch.load("torch-saved-PurdueShapes5-label-map.pt")
                    # reverse the key-value pairs in the label dictionary:
                    self.class_labels = dict(map(reversed, self.label_map.items()))
                    self.transform = transform
                else:
                    print("""\n\n\nLooks like this is the first time you will be loading in\n"""
                          """the dataset for this script. First time loading could take\n"""
                          """a minute or so.  Any subsequent attempts will only take\n"""
                          """a few seconds.\n\n\n""")
                    root_dir = dl_studio.dataroot
                    f = gzip.open(root_dir + dataset_file, 'rb')
                    dataset = f.read()
                    if sys.version_info[0] == 3:
                        self.dataset, self.label_map = pickle.loads(dataset, encoding='latin1')
                    else:
                        self.dataset, self.label_map = pickle.loads(dataset)
                    torch.save(self.dataset, "torch-saved-PurdueShapes5-10000-dataset-noise-20.pt")
                    torch.save(self.label_map, "torch-saved-PurdueShapes5-label-map.pt")
                    # reverse the key-value pairs in the label dictionary:
                    self.class_labels = dict(map(reversed, self.label_map.items()))
                    self.transform = transform
            elif train_or_test == 'train' and dataset_file == "PurdueShapes5-10000-train-noise-50.gz":
                if os.path.exists("torch-saved-PurdueShapes5-10000-dataset-noise-50.pt") and \
                        os.path.exists("torch-saved-PurdueShapes5-label-map.pt"):
                    print("\nLoading training data from the torch-saved archive")
                    self.dataset = torch.load("torch-saved-PurdueShapes5-10000-dataset-noise-50.pt")
                    self.label_map = torch.load("torch-saved-PurdueShapes5-label-map.pt")
                    # reverse the key-value pairs in the label dictionary:
                    self.class_labels = dict(map(reversed, self.label_map.items()))
                    self.transform = transform
                else:
                    print("""\n\n\nLooks like this is the first time you will be loading in\n"""
                          """the dataset for this script. First time loading could take\n"""
                          """a minute or so.  Any subsequent attempts will only take\n"""
                          """a few seconds.\n\n\n""")
                    root_dir = dl_studio.dataroot
                    f = gzip.open(root_dir + dataset_file, 'rb')
                    dataset = f.read()
                    if sys.version_info[0] == 3:
                        self.dataset, self.label_map = pickle.loads(dataset, encoding='latin1')
                    else:
                        self.dataset, self.label_map = pickle.loads(dataset)
                    torch.save(self.dataset, "torch-saved-PurdueShapes5-10000-dataset-noise-50.pt")
                    torch.save(self.label_map, "torch-saved-PurdueShapes5-label-map.pt")
                    # reverse the key-value pairs in the label dictionary:
                    self.class_labels = dict(map(reversed, self.label_map.items()))
                    self.transform = transform
            elif train_or_test == 'train' and dataset_file == "PurdueShapes5-10000-train-noise-80.gz":
                if os.path.exists("torch-saved-PurdueShapes5-10000-dataset-noise-80.pt") and \
                        os.path.exists("torch-saved-PurdueShapes5-label-map.pt"):
                    print("\nLoading training data from the torch-saved archive")
                    self.dataset = torch.load("torch-saved-PurdueShapes5-10000-dataset-noise-80.pt")
                    self.label_map = torch.load("torch-saved-PurdueShapes5-label-map.pt")
                    # reverse the key-value pairs in the label dictionary:
                    self.class_labels = dict(map(reversed, self.label_map.items()))
                    self.transform = transform
                else:
                    print("""\n\n\nLooks like this is the first time you will be loading in\n"""
                          """the dataset for this script. First time loading could take\n"""
                          """a minute or so.  Any subsequent attempts will only take\n"""
                          """a few seconds.\n\n\n""")
                    root_dir = dl_studio.dataroot
                    f = gzip.open(root_dir + dataset_file, 'rb')
                    dataset = f.read()
                    if sys.version_info[0] == 3:
                        self.dataset, self.label_map = pickle.loads(dataset, encoding='latin1')
                    else:
                        self.dataset, self.label_map = pickle.loads(dataset)
                    torch.save(self.dataset, "torch-saved-PurdueShapes5-10000-dataset-noise-80.pt")
                    torch.save(self.label_map, "torch-saved-PurdueShapes5-label-map.pt")
                    # reverse the key-value pairs in the label dictionary:
                    self.class_labels = dict(map(reversed, self.label_map.items()))
                    self.transform = transform
            else:
                root_dir = dl_studio.dataroot
                f = gzip.open(root_dir + dataset_file, 'rb')
                dataset = f.read()
                if sys.version_info[0] == 3:
                    self.dataset, self.label_map = pickle.loads(dataset, encoding='latin1')
                else:
                    self.dataset, self.label_map = pickle.loads(dataset)
                # reverse the key-value pairs in the label dictionary:
                self.class_labels = dict(map(reversed, self.label_map.items()))
                self.transform = transform

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, idx):
            r = np.array(self.dataset[idx][0])
            g = np.array(self.dataset[idx][1])
            b = np.array(self.dataset[idx][2])
            R, G, B = r.reshape(32, 32), g.reshape(32, 32), b.reshape(32, 32)
            im_tensor = torch.zeros(3, 32, 32, dtype=torch.float)
            im_tensor[0, :, :] = torch.from_numpy(R)
            im_tensor[1, :, :] = torch.from_numpy(G)
            im_tensor[2, :, :] = torch.from_numpy(B)
            bb_tensor = torch.tensor(self.dataset[idx][3], dtype=torch.float)
            sample = {'image': im_tensor,
                      'bbox': bb_tensor,
                      'label': self.dataset[idx][4]}
            if self.transform:
                sample = self.transform(sample)
            return sample

    def load_PurdueShapes5_dataset(self, dataserver_train, dataserver_test):
        #            transform = tvt.Compose([tvt.ToTensor(),
        #                                tvt.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.train_dataloader = torch.utils.data.DataLoader(dataserver_train,
                                                            batch_size=self.dl_studio.batch_size, shuffle=True,
                                                            num_workers=4)
        self.test_dataloader = torch.utils.data.DataLoader(dataserver_test,
                                                           batch_size=self.dl_studio.batch_size, shuffle=False,
                                                           num_workers=4)

    class SkipBlock(nn.Module):
        def __init__(self, in_ch, out_ch, downsample=False, skip_connections=True):
            super(DetectAndLocalize.SkipBlock, self).__init__()
            self.downsample = downsample
            self.skip_connections = skip_connections
            self.in_ch = in_ch
            self.out_ch = out_ch
            self.convo = nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1)
            norm_layer = nn.BatchNorm2d
            self.bn = norm_layer(out_ch)
            if downsample:
                self.downsampler = nn.Conv2d(in_ch, out_ch, 1, stride=2)

        def forward(self, x):
            identity = x
            out = self.convo(x)
            out = self.bn(out)
            out = torch.nn.functional.relu(out)
            if self.in_ch == self.out_ch:
                out = self.convo(out)
                out = self.bn(out)
                out = torch.nn.functional.relu(out)
            if self.downsample:
                out = self.downsampler(out)
                identity = self.downsampler(identity)
            if self.skip_connections:
                if self.in_ch == self.out_ch:
                    out += identity
                else:
                    out[:, :self.in_ch, :, :] += identity
                    out[:, self.in_ch:, :, :] += identity
            return out

    class LOADnet2(nn.Module):
        """
            The acronym 'LOAD' stands for 'LOcalization And Detection'.
            LOADnet2 uses both convo and linear layers for regression
            """

        def __init__(self, skip_connections=True, depth=32):
            super(DetectAndLocalize.LOADnet2, self).__init__()
            self.pool_count = 3
            self.mu_x = torch.nn.Parameter(torch.tensor([0.]))
            self.mu_y = torch.nn.Parameter(torch.tensor([0.]))
            self.sigma_x = torch.nn.Parameter(torch.tensor([0.5]))
            self.sigma_y = torch.nn.Parameter(torch.tensor([0.5]))
            self.mu_x.requires_grad = True
            self.mu_y.requires_grad = True
            self.sigma_x.requires_grad = True
            self.sigma_y.requires_grad = True

            self.depth = depth // 2
            self.conv = nn.Conv2d(3, 64, 3, padding=1)
            self.pool = nn.MaxPool2d(2, 2)
            self.skip64 = DetectAndLocalize.SkipBlock(64, 64,
                                                      skip_connections=skip_connections)
            self.skip64ds = DetectAndLocalize.SkipBlock(64, 64,
                                                        downsample=True, skip_connections=skip_connections)
            self.skip64to128 = DetectAndLocalize.SkipBlock(64, 128,
                                                           skip_connections=skip_connections)
            self.skip128 = DetectAndLocalize.SkipBlock(128, 128,
                                                       skip_connections=skip_connections)
            self.skip128ds = DetectAndLocalize.SkipBlock(128, 128,
                                                         downsample=True, skip_connections=skip_connections)
            self.fc1 = nn.Linear(128 * (32 // 2 ** self.pool_count) ** 2, 1000)
            self.fc2 = nn.Linear(1000, 5)
            ##  for regression
            self.conv_seqn = nn.Sequential(
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )
            self.fc_seqn = nn.Sequential(
                nn.Linear(16384, 1024),
                nn.ReLU(inplace=True),
                nn.Linear(1024, 512),
                nn.ReLU(inplace=True),
                nn.Linear(512, 4)
            )

        def forward(self, x3):
            kernel = gaussian_filter(3, self.mu_x, self.mu_y, self.sigma_x, self.sigma_y)
            kernel = kernel.to(device)

            x = apply_filter(x3, kernel)

            x = self.pool(torch.nn.functional.relu(self.conv(x)))
            ## The labeling section:
            x1 = x.clone()
            for _ in range(self.depth // 4):
                x1 = self.skip64(x1)
            x1 = self.skip64ds(x1)
            for _ in range(self.depth // 4):
                x1 = self.skip64(x1)
            x1 = self.skip64to128(x1)
            for _ in range(self.depth // 4):
                x1 = self.skip128(x1)
            x1 = self.skip128ds(x1)
            for _ in range(self.depth // 4):
                x1 = self.skip128(x1)
            x1 = x1.view(-1, 128 * (32 // 2 ** self.pool_count) ** 2)
            x1 = torch.nn.functional.relu(self.fc1(x1))
            x1 = self.fc2(x1)
            ## The Bounding Box regression:
            x2 = self.conv_seqn(x)
            x2 = self.conv_seqn(x2)
            # flatten
            x2 = x2.view(x.size(0), -1)
            x2 = self.fc_seqn(x2)
            return x1, x2

    def run_code_for_training_with_CrossEntropy_and_MSE_Losses(self, net):
        filename_for_out1 = "performance_numbers_" + str(self.dl_studio.epochs) + "label.txt"
        filename_for_out2 = "performance_numbers_" + str(self.dl_studio.epochs) + "regres.txt"
        FILE1 = open(filename_for_out1, 'w')
        FILE2 = open(filename_for_out2, 'w')
        net = copy.deepcopy(net)
        net = net.to(self.dl_studio.device)
        criterion1 = nn.CrossEntropyLoss()
        criterion2 = nn.MSELoss()
        optimizer = optim.SGD(net.parameters(),
                              lr=self.dl_studio.learning_rate, momentum=self.dl_studio.momentum)
        for epoch in range(self.dl_studio.epochs):
            running_loss_labeling = 0.0
            running_loss_regression = 0.0
            for i, data in enumerate(self.train_dataloader):
                gt_too_small = False
                inputs, bbox_gt, labels = data['image'], data['bbox'], data['label']
                if self.dl_studio.debug_train and i % 500 == 499:
                    #                    if self.dl_studio.debug_train and ((epoch==0 and (i==0 or i==9 or i==99)) or i%500==499):
                    print("\n\n[epoch=%d iter=%d:] Ground Truth:     " % (epoch + 1, i + 1) +
                          ' '.join('%10s' % self.dataserver_train.class_labels[labels[j].item()] for j in
                                   range(self.dl_studio.batch_size)))
                inputs = inputs.to(self.dl_studio.device)
                labels = labels.to(self.dl_studio.device)
                bbox_gt = bbox_gt.to(self.dl_studio.device)
                optimizer.zero_grad()
                outputs = net(inputs)
                outputs_label = outputs[0]
                bbox_pred = outputs[1]
                if self.dl_studio.debug_train and i % 500 == 499:
                    #                  if self.dl_studio.debug_train and ((epoch==0 and (i==0 or i==9 or i==99)) or i%500==499):
                    inputs_copy = inputs.detach().clone()
                    inputs_copy = inputs_copy.cpu()
                    bbox_pc = bbox_pred.detach().clone()
                    bbox_pc[bbox_pc < 0] = 0
                    bbox_pc[bbox_pc > 31] = 31
                    bbox_pc[torch.isnan(bbox_pc)] = 0
                    _, predicted = torch.max(outputs_label.data, 1)
                    print("[epoch=%d iter=%d:] Predicted Labels: " % (epoch + 1, i + 1) +
                          ' '.join('%10s' % self.dataserver_train.class_labels[predicted[j].item()]
                                   for j in range(self.dl_studio.batch_size)))
                    for idx in range(self.dl_studio.batch_size):
                        i1 = int(bbox_gt[idx][1])
                        i2 = int(bbox_gt[idx][3])
                        j1 = int(bbox_gt[idx][0])
                        j2 = int(bbox_gt[idx][2])
                        k1 = int(bbox_pc[idx][1])
                        k2 = int(bbox_pc[idx][3])
                        l1 = int(bbox_pc[idx][0])
                        l2 = int(bbox_pc[idx][2])
                        print("                    gt_bb:  [%d,%d,%d,%d]" % (j1, i1, j2, i2))
                        print("                  pred_bb:  [%d,%d,%d,%d]" % (l1, k1, l2, k2))
                        inputs_copy[idx, 0, i1:i2, j1] = 255
                        inputs_copy[idx, 0, i1:i2, j2] = 255
                        inputs_copy[idx, 0, i1, j1:j2] = 255
                        inputs_copy[idx, 0, i2, j1:j2] = 255
                        inputs_copy[idx, 2, k1:k2, l1] = 255
                        inputs_copy[idx, 2, k1:k2, l2] = 255
                        inputs_copy[idx, 2, k1, l1:l2] = 255
                        inputs_copy[idx, 2, k2, l1:l2] = 255
                #                        self.dl_studio.display_tensor_as_image(
                #                              torchvision.utils.make_grid(inputs_copy, normalize=True),
                #                             "see terminal for TRAINING results at iter=%d" % (i+1))
                loss_labeling = criterion1(outputs_label, labels)
                loss_labeling.backward(retain_graph=True)
                loss_regression = criterion2(bbox_pred, bbox_gt)
                loss_regression.backward()
                optimizer.step()
                running_loss_labeling += loss_labeling.item()
                running_loss_regression += loss_regression.item()
                if i % 500 == 499:
                    avg_loss_labeling = running_loss_labeling / float(500)
                    avg_loss_regression = running_loss_regression / float(500)
                    print("\n[epoch:%d, iteration:%5d]  loss_labeling: %.3f  loss_regression: %.3f  " % (
                        epoch + 1, i + 1, avg_loss_labeling, avg_loss_regression))
                    FILE1.write("%.3f\n" % avg_loss_labeling)
                    FILE1.flush()
                    FILE2.write("%.3f\n" % avg_loss_regression)
                    FILE2.flush()
                    running_loss_labeling = 0.0
                    running_loss_regression = 0.0
        #                    if self.dl_studio.debug_train and i % 500 == 499:
        #                    if self.dl_studio.debug_train and ((epoch==0 and (i==0 or i==9 or i==99)) or i%500==499):
        #                        self.dl_studio.display_tensor_as_image(
        #                            torchvision.utils.make_grid(inputs_copy, normalize=True),
        #                            "see terminal for TRAINING results at iter=%d" % (i + 1))

        print("\nFinished Training\n")
        self.save_model(net)

    def save_model(self, model):
        '''
            Save the trained model to a disk file
            '''
        torch.save(model.state_dict(), self.dl_studio.path_saved_model)

    def run_code_for_testing_detection_and_localization(self, net,file):
        net.load_state_dict(torch.load(self.dl_studio.path_saved_model))
        correct = 0
        total = 0
        confusion_matrix = torch.zeros(len(self.dataserver_train.class_labels),
                                       len(self.dataserver_train.class_labels))
        class_correct = [0] * len(self.dataserver_train.class_labels)
        class_total = [0] * len(self.dataserver_train.class_labels)
        with torch.no_grad():
            for i, data in enumerate(self.test_dataloader):
                images, bounding_box, labels = data['image'], data['bbox'], data['label']
                images = images.to(self.dl_studio.device)
                labels = labels.tolist()
                if self.dl_studio.debug_test and i % 50 == 0:
                    print("\n\n[i=%d:] Ground Truth:     " % i + ' '.join('%10s' %
                                                                          self.dataserver_train.class_labels[
                                                                              labels[j]] for j in
                                                                          range(self.dl_studio.batch_size)))
                outputs = net(images)
                outputs_label = outputs[0]
                outputs_regression = outputs[1]
                outputs_regression[outputs_regression < 0] = 0
                outputs_regression[outputs_regression > 31] = 31
                outputs_regression[torch.isnan(outputs_regression)] = 0
                output_bb = outputs_regression.tolist()
                _, predicted = torch.max(outputs_label.data, 1)
                predicted = predicted.tolist()
                if self.dl_studio.debug_test and i % 50 == 0:
                    print("[i=%d:] Predicted Labels: " % i + ' '.join('%10s' %
                                                                      self.dataserver_train.class_labels[
                                                                          predicted[j]] for j in
                                                                      range(self.dl_studio.batch_size)))
                    for idx in range(self.dl_studio.batch_size):
                        i1 = int(bounding_box[idx][1])
                        i2 = int(bounding_box[idx][3])
                        j1 = int(bounding_box[idx][0])
                        j2 = int(bounding_box[idx][2])
                        k1 = int(output_bb[idx][1])
                        k2 = int(output_bb[idx][3])
                        l1 = int(output_bb[idx][0])
                        l2 = int(output_bb[idx][2])
                        print("                    gt_bb:  [%d,%d,%d,%d]" % (j1, i1, j2, i2))
                        print("                  pred_bb:  [%d,%d,%d,%d]" % (l1, k1, l2, k2))
                        images[idx, 0, i1:i2, j1] = 255
                        images[idx, 0, i1:i2, j2] = 255
                        images[idx, 0, i1, j1:j2] = 255
                        images[idx, 0, i2, j1:j2] = 255
                        images[idx, 2, k1:k2, l1] = 255
                        images[idx, 2, k1:k2, l2] = 255
                        images[idx, 2, k1, l1:l2] = 255
                        images[idx, 2, k2, l1:l2] = 255
                #                       self.dl_studio.display_tensor_as_image(
                #                            torchvision.utils.make_grid(images, normalize=True),
                #                           "see terminal for test results at i=%d" % i)
                for label, prediction in zip(labels, predicted):
                    confusion_matrix[label][prediction] += 1
                total += len(labels)
                correct += [predicted[ele] == labels[ele] for ele in range(len(predicted))].count(True)
                comp = [predicted[ele] == labels[ele] for ele in range(len(predicted))]
                for j in range(self.dl_studio.batch_size):
                    label = labels[j]
                    class_correct[label] += comp[j]
                    class_total[label] += 1
        print("\n")
        for j in range(len(self.dataserver_train.class_labels)):
            print('Prediction accuracy for %5s : %2d %%' % (
                self.dataserver_train.class_labels[j], 100 * class_correct[j] / class_total[j]))
        print("\n\n\nOverall accuracy of the network on the 1000 test images: %d %%" %
              (100 * correct / float(total)))
        file.write("\n\n\nOverall accuracy of the network on the 1000 test images: %s %%" %
              (str(100 * correct / float(total))))
        print("\n\nDisplaying the confusion matrix:\n")
        out_str = "                "
        for j in range(len(self.dataserver_train.class_labels)):
            out_str += "%15s" % self.dataserver_train.class_labels[j]
        print(out_str + "\n")
        for i, label in enumerate(self.dataserver_train.class_labels):
            out_percents = [100 * confusion_matrix[i, j] / float(class_total[i])
                            for j in range(len(self.dataserver_train.class_labels))]
            out_percents = ["%.2f" % item.item() for item in out_percents]
            out_str = "%12s:  " % self.dataserver_train.class_labels[i]
            for j in range(len(self.dataserver_train.class_labels)):
                out_str += "%15s" % out_percents[j]
            print(out_str)


##  watch -d -n 0.5 nvidia-smi

dls = DLStudio(
    dataroot="./data/",
    image_size=[32, 32],
    path_saved_model="./saved_model",
    momentum=0.9,
    learning_rate=5e-5,
    epochs=5,
    batch_size=4,
    classes=('rectangle', 'triangle', 'disk', 'oval', 'star'),
    debug_train=1,
    debug_test=1,
    use_gpu=True,
)

train_paths = ["PurdueShapes5-10000-train.gz",
               "PurdueShapes5-10000-train-noise-20.gz",
               "PurdueShapes5-10000-train-noise-50.gz",
               "PurdueShapes5-10000-train-noise-80.gz"
               ]
test_paths = ["PurdueShapes5-1000-test.gz",
              "PurdueShapes5-1000-test-noise-20.gz",
              "PurdueShapes5-1000-test-noise-50.gz",
              "PurdueShapes5-1000-test-noise-80.gz"
              ]
file = open('gaussian_values.txt', 'w+')
for train_path, test_path in zip(train_paths, test_paths):
    detector = DetectAndLocalize(dl_studio=dls)
    dataserver_train = DetectAndLocalize.PurdueShapes5Dataset(
        train_or_test='train',
        dl_studio=dls,
        # dataset_file = "PurdueShapes5-20-train.gz"
        dataset_file=train_path
    )
    dataserver_test = DetectAndLocalize.PurdueShapes5Dataset(
        train_or_test='test',
        dl_studio=dls,
        # dataset_file = "PurdueShapes5-20-test.gz"
        dataset_file=test_path
    )
    detector.dataserver_train = dataserver_train
    detector.dataserver_test = dataserver_test

    detector.load_PurdueShapes5_dataset(dataserver_train, dataserver_test)

    model = detector.LOADnet2(skip_connections=True, depth=32)
    model.cuda()
    detector.run_code_for_training_with_CrossEntropy_and_MSE_Losses(model)
    # detector.run_code_for_training_with_CrossEntropy_and_BCE_Losses(model)

    import pymsgbox

    detector.run_code_for_testing_detection_and_localization(model,file)
    print(model.mu_x, model.mu_y, model.sigma_x, model.sigma_y)
    file.write("mu_x=%s, mu_y=%s, sigma_x=%s,sigma_y=%s\n\n" % (
    str(model.mu_x.item()), str(model.mu_y.item()), str(model.sigma_x.item()), str(model.sigma_y.item())))
file.close()

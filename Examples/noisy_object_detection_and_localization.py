#!/usr/bin/env python

##  noisy_object_detection_and_localization.py

"""
This script in the Examples directory does exactly the same thing as the following
script
           object_detection_and_localization.py

with the the only difference being that the "noisy" version of the script here calls
on the noise-corrupted training and testing dataset files.  I thought it would be
best to create a separate script for studying the effects of noise, just to allow for
the possibility the noise-related studies may evolve differently in the future.

As is the case with 'object_detection_and_localization.py', this script shows how you
can use the functionality provided by the inner class DetectAndLocalize of the
DLStudio module for experimenting with object detection and localization.

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

import random
import numpy
import torch
import os, sys

seed = 0
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
numpy.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmarks = False
os.environ['PYTHONHASHSEED'] = str(seed)
sys.path.append("/home/raghav0/dl/DLStudio-1.1.0")

##  watch -d -n 0.5 nvidia-smi

from DLStudio import *

file = open('output.txt', 'w')

# loop parameters
learning_rates = [5e-6,1e-5, 5e-5, 8e-5, 1e-4]

filters_list = [{'filter': None, 'sigma': 1},
                {'filter': 'mean', 'sigma': 1},
                {'filter': 'gaussian', 'sigma': 0.5},
                {'filter': 'gaussian', 'sigma': 1},
                {'filter': 'gaussian', 'sigma': 5},
                {'filter': 'gaussian', 'sigma': 10},
                {'filter': 'gaussian', 'sigma': 50}]


noise_levels = ['20','50','80']
file.write("Epochs : 5\n\n")
for rate in learning_rates:
    dls = DLStudio(
        dataroot="./data/",
        image_size=[32, 32],
        path_saved_model="./saved_model",
        momentum=0.9,
        learning_rate=rate,
        epochs=5,
        batch_size=4,
        classes=('rectangle', 'triangle', 'disk', 'oval', 'star'),
        debug_train=1,
        debug_test=1,
        use_gpu=True,
    )

    detector = DLStudio.DetectAndLocalize(dl_studio=dls)

    for filter_element in filters_list:
        for noise in noise_levels:
            dataserver_train = DLStudio.DetectAndLocalize.PurdueShapes5Dataset(
                train_or_test='train',
                dl_studio=dls,
                filter=filter_element['filter'],
                sigma=filter_element['sigma'],
                #                                   dataset_file = "PurdueShapes5-20-train.gz",
                # dataset_file = "PurdueShapes5-10000-train-noise-20.gz",
                # dataset_file="PurdueShapes5-10000-train-noise-50.gz",
                dataset_file="PurdueShapes5-10000-train-noise-" + noise + ".gz"
                #                                   dataset_file = "PurdueShapes5-10000-train-noise-80.gz",
            )
            dataserver_test = DLStudio.DetectAndLocalize.PurdueShapes5Dataset(
                train_or_test='test',
                dl_studio=dls,
                filter=filter_element['filter'],
                sigma=filter_element['sigma'],
                #                                   dataset_file = "PurdueShapes5-20-test.gz"
                # dataset_file = "PurdueShapes5-1000-test-noise-20.gz"
                # dataset_file="PurdueShapes5-1000-test-noise-50.gz"
                dataset_file="PurdueShapes5-1000-test-noise-" + noise + ".gz",
                #                                   dataset_file = "PurdueShapes5-1000-test-noise-80.gz"
            )
            detector.dataserver_train = dataserver_train
            detector.dataserver_test = dataserver_test

            detector.load_PurdueShapes5_dataset(dataserver_train, dataserver_test)

            model = detector.LOADnet2(skip_connections=True, depth=32)

            dls.show_network_summary(model)

            detector.run_code_for_training_with_CrossEntropy_and_MSE_Losses(model)
            # detector.run_code_for_training_with_CrossEntropy_and_BCE_Losses(model)

            # import pymsgbox
            #
            # response = pymsgbox.confirm("Finished training.  Start testing on unseen data?")
            # if response == "OK":
            file.write("\n****************************\n")
            file.write("Learning Rate: %s\n" % (str(rate)))
            file.write("Filter : %s and sigma: %s\n" % (filter_element['filter'], str(filter_element['sigma'])))
            file.write("Noise level : %s\n" % noise)

            detector.run_code_for_testing_detection_and_localization(model,file)
            file.write("\n****************************\n")

file.close()


import pandas as pd
import numpy as np
import os
from PIL import Image
from array import *
import keras.backend as K
from sklearn.model_selection import train_test_split
import glob

dirname = 'train'

label_names = ['rotated_left','upright','rotated_right','upside_down']
def conv_label2int(label):
    idx = label_names.index( label )
    return idx

def conv_int2label(index):
    return label_names[index]

def get_image(dirname,file):
    image = np.empty([3,64,64],dtype='uint8')
    imfile = Image.open(os.path.join(dirname, file))
    pixels = imfile.load()
    for color in range(0,3):
        for x in range(0,64):
            for y in range(0,64):
                image[color,x,y] = pixels[x,y][color]

    return image

def load_data():
    """Loads rotfaces dataset.

    # Returns
        Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
    """
    dirname = 'train'

    # Get ground truth file
    train_truth = pd.read_csv("train.truth.csv")

    # Get number of samples
    num_samples = len(train_truth.index)

    # Get file names
    filenames = train_truth[["fn"]].to_numpy()

    # Get images from file names
    x = np.empty((num_samples, 3, 64, 64), dtype='uint8')
    idx = 0
    for fn in filenames:
        x[idx, :, :, :] = get_image(dirname,fn.item())
        idx = idx+1

    # Get labels
    y = train_truth[["label"]].to_numpy()
    vconv = np.vectorize(conv_label2int)
    y = vconv(y)

    ## Train: 75%, Test: 25%
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1/4, 
                                                    random_state=42, stratify=y)


    if K.image_data_format() == 'channels_last':
        x_train = x_train.transpose(0, 2, 3, 1)
        x_test = x_test.transpose(0, 2, 3, 1)

    return (x_train, y_train), (x_test, y_test)


def get_filelist(dirname):
    imgfiles = []
    for filename in glob.glob(dirname+"/*"):
        imgfiles.append(filename.split('/')[1])
    return imgfiles

def load_test_data():
    """Loads rotfaces test dataset to perform predictions.

    # Returns
        Tuple of Numpy arrays: `(x_pred, fn_pred)`.
    """
    dirname = 'test'
    fn_pred = np.asarray(get_filelist(dirname))
    fn_pred = np.reshape(fn_pred, (len(fn_pred), 1))

    x_pred = np.empty((len(fn_pred), 3, 64, 64), dtype='uint8')
    idx = 0
    for filename in fn_pred:
        x_pred[idx, :, :, :] = get_image(dirname,filename.item())
        idx = idx+1

    if K.image_data_format() == 'channels_last':
        x_pred = x_pred.transpose(0, 2, 3, 1)

    return (x_pred, fn_pred)


def rotate_images(fn_pred,y_pred):
    dirname = 'test'
    if not os.path.isdir('preds'):
        os.makedirs('preds')
    idx = 0
    for filename in fn_pred:
        im = Image.open(os.path.join(dirname, filename.item()))
        num_label = conv_label2int( y_pred[idx].item() )
        
        # Angle is in degrees counter clockwise
        if num_label==0: #rotated_left
            im = im.rotate(angle=270)
        elif num_label==2: #rotated_right
            im = im.rotate(angle=90)
        elif num_label==3: #upside_down
            im = im.rotate(angle=180)
        
        im.save(os.path.join('preds',filename.item().split('.')[0]+'.png'))

        idx = idx+1
    
    x_corrected = np.empty((len(fn_pred), 3, 64, 64), dtype='uint8')
    idx = 0
    for filename in fn_pred:
        x_corrected[idx, :, :, :] = get_image('preds',filename.item().split('.')[0]+'.png')
        idx = idx+1
    
    if K.image_data_format() == 'channels_last':
        x_corrected = x_corrected.transpose(0, 2, 3, 1)

    return x_corrected
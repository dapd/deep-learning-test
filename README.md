# deep-learning-test
deep-learning computer vision task - Answer

## Dependencies

* Python 3
* keras
* sklearn
* tensorflow
* numpy
* pandas
* Pillow

## Running

```sh
python3 deeplearning_test.py
```

## Description

Initially, I studied the Keras's CIFAR10 model and tried to understand how the images were formatted in order to provide them as input to the model.

After that, I implemented the 'load_data()' function (rotfaces.py file) to read the images described in the 'train.truth.csv' file and return them with their respective labels. 'load_test_data()' function was implemented to get the test set file names, read the images and return them. 'rotate_images(fn_pred,y_pred)' function was implemented to rotate the images and return them after correction.

Then, I executed the CIFAR10 model with the stop criterion when the 'val_loss' stops improving. It was obtained:

* Test loss: 0.4375; Test accuracy: 0.7293

Finally I made some modifications to the model, removing two Conv2D layers and one MaxPooling2D. It was obtained:

* Test loss: 0.0955; Test accuracy: 0.9686

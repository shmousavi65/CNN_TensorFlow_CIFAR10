# CNN_TensorFlow_CIFAR10
Convolutional Neural Network developed by TensorFlow for CIFAR10 Image Classification

## Notes:
- Dropout and L_2 Regularization methods are incorporated to avoid overfitting in the training process.
- Data manipulation, including image random flip, are used to improve training task
- TensorFlow Dataset class is used to integrate the reading input task, data processing (including image normalization and random flip), and training as a unifies graph.

## Prerequisits
- Python 3.5
- Tensorflow 1.4
- Numpy
- Pickle

## Usage and Execution
- CIFAR10 data should be exracted and and located in the same directory as the codes files. Cifar10 data includes data-batch_1, data-batch_2, data-batch_3, data-batch_4, data-batch_5, and test-batch  
- Change the directory to the codes and data files directory:
```
    cd CNN_TensorFlow_CIFAR10
```
- run the main file:  
``` 
   python3  main.py
```
 


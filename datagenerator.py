import numpy as np
import cv2
import sys
import copy

sys.path.append('/home/zxz/fashion-mnist')
from utils import mnist_reader

"""
This code is highly influenced by the implementation of:
https://github.com/joelthchao/tensorflow-finetune-flickr-style/dataset.py
But changed abit to allow dataaugmentation (yet only horizontal flip) and 
shuffling of the data. 
The other source of inspiration is the ImageDataGenerator by @fchollet in the 
Keras library. But as I needed BGR color format for fine-tuneing AlexNet I 
wrote my own little generator.
"""

class ImageDataGenerator:
    def __init__(self, phase='TRAIN', horizontal_flip=False, shuffle=False, 
                 scale_size = [28, 28], nb_classes = 10):
        
                
        # Init params
        self.horizontal_flip = horizontal_flip
        self.n_classes = nb_classes
        self.shuffle = shuffle
        self.pointer = 0
        self.phase = phase
        self.scale_size = scale_size
        self.read_data()
        self.data_size = self.images.shape[0]

        print "data size of phase {} is {}\n".format(phase, self.data_size)

        if self.shuffle:
            self.shuffle_data()

    def read_data(self):
        if self.phase == 'TRAIN':
            self.images, self.labels = mnist_reader.load_mnist('/home/zxz/fashion-mnist/data/fashion', kind='train')
        elif self.phase == 'TEST':
            self.images, self.labels = mnist_reader.load_mnist('/home/zxz/fashion-mnist/data/fashion', kind='t10k')
        else:
            raise Exception('invalid phase: ', self.phase)


    def shuffle_data(self):
        """
        Random shuffle the images and labels
        """
        images = copy.deepcopy(self.images)
        labels = copy.deepcopy(self.labels)
        self.images = []
        self.labels = []
        
        #create list of permutated index and shuffle data accoding to list
        idx = np.random.permutation(len(labels))
        for i in idx:
            self.images.append(images[i])
            self.labels.append(labels[i])
                
    def reset_pointer(self):
        """
        reset pointer to begin of the list
        """
        self.pointer = 0
        
        if self.shuffle:
            self.shuffle_data()
        
    
    def next_batch(self, batch_size):
        """
        This function gets the next n ( = batch_size) images from the path list
        and labels and loads the images into them into memory 
        """
        # Get next batch of image (path) and labels
        raw_images = self.images[self.pointer:self.pointer + batch_size]
        labels = self.labels[self.pointer:self.pointer + batch_size]
        images = np.ndarray([batch_size, self.scale_size[0], self.scale_size[1], 1])

        #update pointer
        self.pointer += batch_size
        
        for i in range(batch_size):
            images[i] = self.images[i].reshape([self.scale_size[0], self.scale_size[1], 1])
        
        # Expand labels to one hot encoding
        one_hot_labels = np.zeros((batch_size, self.n_classes))
        for i in range(len(labels)):
            one_hot_labels[i][labels[i]] = 1

        #return array of images and labels
        return images, one_hot_labels

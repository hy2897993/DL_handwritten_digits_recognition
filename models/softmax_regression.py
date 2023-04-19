""" 			  		 			     			  	   		   	  			  	
Softmax Regression Model.  (c) 2021 Georgia Tech

Copyright 2021, Georgia Institute of Technology (Georgia Tech)
Atlanta, Georgia 30332
All Rights Reserved

Template code for CS 7643 Deep Learning

Georgia Tech asserts copyright ownership of this template and all derivative
works, including solutions to the projects assigned in this course. Students
and other users of this template code are advised not to share it with others
or to make it available on publicly viewable websites including repositories
such as Github, Bitbucket, and Gitlab.  This copyright statement should
not be removed or edited.

Sharing solutions with current or future students of CS 7643 Deep Learning is
prohibited and subject to being investigated as a GT honor code violation.

-----do not edit anything above this line---
"""

# Do not use packages that are not in standard distribution of python
import numpy as np

from ._base_network import _baseNetwork


class SoftmaxRegression(_baseNetwork):
    def __init__(self, input_size=28 * 28, num_classes=10):
        """
        A single layer softmax regression. The network is composed by:
        a linear layer without bias => (activation) => Softmax
        :param input_size: the input dimension
        :param num_classes: the number of classes in total
        """
        super().__init__(input_size, num_classes)
        self._weight_init()

    def _weight_init(self):
        '''
        initialize weights of the single layer regression network. No bias term included.
        :return: None; self.weights is filled based on method
        - W1: The weight matrix of the linear layer of shape (num_features, hidden_size)
        '''
        np.random.seed(1024)
        self.weights['W1'] = 0.001 * np.random.randn(self.input_size, self.num_classes)
        self.gradients['W1'] = np.zeros((self.input_size, self.num_classes))

    def forward(self, X, y, mode='train'):
        """
        Compute loss and gradients using softmax with vectorization.

        :param X: a batch of image (N, 28x28)
        :param y: labels of images in the batch (N,)
        :return:
            loss: the loss associated with the batch
            accuracy: the accuracy of the batch
        """
        loss = 0
        gradient = np.zeros((self.input_size, self.num_classes))
        accuracy = 0
        accu_count = 0
        n = len(y)
        
        #############################################################################
        # TODO:                                                                     #
        #    1) Implement the forward process and compute the Cross-Entropy loss    #
        #    2) Compute the gradient of the loss with respect to the weights        #
        # Hint:                                                                     #
        #   Store your intermediate outputs before ReLU for backwards               #
        #############################################################################
        for input, label in zip(X, y):
            layerone = np.expand_dims(input,axis = 1)*self.weights['W1']
            layerone_output = np.sum(layerone, axis = 0)
            relus = np.array([max(a,0) for a in layerone_output])
            zero = np.array([a>0 for a in layerone_output])

  
            
            explist = np.exp(relus)
            expsum = sum(explist)
            softmax = explist/expsum

            L = -np.log(softmax[label])
            loss += L
            
            if mode == 'train':
                gradient_softmax = (softmax - np.array([int(cl == label) for cl in range(self.num_classes)]))
                gradient += np.expand_dims(input,axis = 1) * gradient_softmax * np.expand_dims(zero,axis = 0) 
                

                # self.gradients['W1'] += np.expand_dims(input,axis = 1) * gradient_softmax * np.expand_dims(zero,axis = 0) 
                
            
            if np.argmax(relus) == label:
                accu_count += 1
        
        loss = loss/n
        
        accuracy = accu_count/n


            
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        if mode != 'train':
            return loss, accuracy

        #############################################################################
        # TODO:                                                                     #
        #    1) Implement the backward process:                                     #
        #        1) Compute gradients of each weight by chain rule                  #
        #        2) Store the gradients in self.gradients                           #
        #############################################################################
        self.gradients['W1'] += gradient/n
                    
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return loss, accuracy

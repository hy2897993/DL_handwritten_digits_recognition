""" 			  		 			     			  	   		   	  			  	
MLP Model.  (c) 2021 Georgia Tech

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

np.random.seed(1024)
from ._base_network import _baseNetwork


class TwoLayerNet(_baseNetwork):
    def __init__(self, input_size=28 * 28, num_classes=10, hidden_size=128):
        super().__init__(input_size, num_classes)

        self.hidden_size = hidden_size
        self._weight_init()

    def _weight_init(self):
        """
        initialize weights of the network
        :return: None; self.weights is filled based on method
        - W1: The weight matrix of the first layer of shape (num_features, hidden_size)
        - b1: The bias term of the first layer of shape (hidden_size,)
        - W2: The weight matrix of the second layer of shape (hidden_size, num_classes)
        - b2: The bias term of the second layer of shape (num_classes,)
        """

        # initialize weights
        self.weights['b1'] = np.zeros(self.hidden_size)
        self.weights['b2'] = np.zeros(self.num_classes)
        np.random.seed(1024)
        self.weights['W1'] = 0.001 * np.random.randn(self.input_size, self.hidden_size)
        np.random.seed(1024)
        self.weights['W2'] = 0.001 * np.random.randn(self.hidden_size, self.num_classes)

        # initialize gradients to zeros
        self.gradients['W1'] = np.zeros((self.input_size, self.hidden_size))
        self.gradients['b1'] = np.zeros(self.hidden_size)
        self.gradients['W2'] = np.zeros((self.hidden_size, self.num_classes))
        self.gradients['b2'] = np.zeros(self.num_classes)

    def forward(self, X, y, mode='train'):
        """
        The forward pass of the two-layer net. The activation function used in between the two layers is sigmoid, which
        is to be implemented in self.,sigmoid.
        The method forward should compute the loss of input batch X and gradients of each weights.
        Further, it should also compute the accuracy of given batch. The loss and
        accuracy are returned by the method and gradients are stored in self.gradients

        :param X: a batch of images (N, input_size)
        :param y: labels of images in the batch (N,)
        :param mode: if mode is training, compute and update gradients;else, just return the loss and accuracy
        :return:
            loss: the loss associated with the batch
            accuracy: the accuracy of the batch
            self.gradients: gradients are not explicitly returned but rather updated in the class member self.gradients
        """
        loss = 0
        accuracy = 0
        accu_count = 0
        gradient = {}
        gradient["W1"] = np.zeros((self.input_size, self.hidden_size))
        gradient["b1"] = np.zeros(self.hidden_size)
        gradient["W2"] = np.zeros((self.hidden_size, self.num_classes))
        gradient["b2"] = np.zeros(self.num_classes)
        n = len(y)
        #############################################################################
        # TODO:                                                                     #
        #    1) Implement the forward process:                                      #
        #        1) Call sigmoid function between the two layers for non-linearity  #
        #        2) The output of the second layer should be passed to softmax      #
        #        function before computing the cross entropy loss                   #
        #    2) Compute Cross-Entropy Loss and batch accuracy based on network      #
        #       outputs                                                             #
        #############################################################################
        for input, label in zip(X, y):

            sigmoid =1/(1+np.exp(-(np.sum(np.sum(np.expand_dims(input,axis = 1)*self.weights['W1'], axis = 0) + np.expand_dims(self.weights['b1'],axis=0), axis = 0))))
            layerone_output = np.sum(np.sum(np.expand_dims(sigmoid,axis = 1)*self.weights['W2'], axis = 0)+ np.expand_dims(self.weights['b2'],axis=0)
            , axis = 0)

            explist = np.exp(layerone_output)

            expsum = sum(explist)
            softmax = explist/expsum

            L = -np.log(softmax[label])
            loss += L


            if mode == 'train':
                gradient_softmax = softmax - np.array([int(cl == label) for cl in range(self.num_classes)])
                gradient['W2'] += np.expand_dims( sigmoid,axis = 1)*gradient_softmax
                gradient['b2'] += gradient_softmax

                gradient_siged =np.sum(self.weights['W2'] * np.expand_dims(gradient_softmax,axis = 0), axis = 1) 

                gradient_sigmoid = sigmoid*(1-sigmoid)*gradient_siged

                gradient['W1'] += np.expand_dims(input,axis = 1) *gradient_sigmoid

                gradient['b1'] += gradient_sigmoid


            
            if np.argmax(softmax) == label:
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
        #        1) Compute gradients of each weight and bias by chain rule         #
        #        2) Store the gradients in self.gradients                           #
        #    HINT: You will need to compute gradients backwards, i.e, compute       #
        #          gradients of W2 and b2 first, then compute it for W1 and b1      #
        #          You may also want to implement the analytical derivative of      #
        #          the sigmoid function in self.sigmoid_dev first                   #
        #############################################################################
        # for i in range(n):
        #     for c in range(self.num_classes):
        #         self.gradients['b2'][c] += gradient['b2'][i][c]/n 
        #         for hl in range(self.hidden_size):
        #             self.gradients['W2'][hl][c] += gradient['W2'][i][hl][c]/n 
                        
        #     for hl in range(self.hidden_size):
        #         self.gradients['b1'][hl] += gradient['b1'][i][hl]/n 
        #         for idx in range(self.input_size):
        #             self.gradients['W1'][idx][hl] += gradient['W1'][i][idx][hl]/n 
        self.gradients["W1"] += gradient["W1"]/n
        self.gradients["W2"] += gradient["W2"]/n
        self.gradients["b1"] += gradient["b1"]/n
        self.gradients["b2"] += gradient["b2"]/n              



        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

        return loss, accuracy

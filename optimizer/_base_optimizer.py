""" 			  		 			     			  	   		   	  			  	
Optimizer base.  (c) 2021 Georgia Tech

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


class _BaseOptimizer:
    def __init__(self, learning_rate=1e-4, reg=1e-3):
        self.learning_rate = learning_rate
        self.reg = reg

    def update(self, model):
        pass

    def apply_regularization(self, model):
        """
        Apply L2 penalty to the model. Update the gradient dictionary in the model
        :param model: The model with gradients
        :return: None, but the gradient dictionary of the model should be updated
        """

        #############################################################################
        # TODO:                                                                     #
        #    1) Apply L2 penalty to model weights based on the regularization       #
        #       coefficient                                                         #
        #############################################################################
        
        # (self.input_size, self.num_classes)
        
        for key in model.gradients:
            model.gradients[key] += self.reg * model.weights[key]


        # for key in model.weights:
            # model.weights[key] += self.reg * model.gradients[key]

        # for key in model.gradients:
        #     model.gradients[key] += self.reg * model.gradients[key]


        # if "W1" in model.gradients:
        #     for i in range(len(model.gradients["W1"])):
        #         for j in range(len(model.gradients["W1"][i])):
        #             model.gradients["W1"][i][j] += self.reg * model.weights["W1"][i][j] 
        # if "W2" in model.gradients:
        #     for i in range(len(model.gradients["W2"])):
        #         for j in range(len(model.gradients["W2"][i])):
        #             model.gradients["W2"][i][j] += self.reg * model.weights["W2"][i][j] 


        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

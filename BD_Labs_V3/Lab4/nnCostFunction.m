function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%nnCostFunction Implements the neural network cost function for a two layer (L=3)
%neural network which performs classification
%   [J grad] = nnCostFunction(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts. After completing each part, you
%               should verify your result in ex4.m.
%
% Part 4: Feedforward the neural network and return the cost in the
%         variable J. After implementing this part, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 5: Implement regularization with the cost function.
%
% Part 6: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. They are the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 respectively.
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: Backpropagation may be implemented using a for-loop
%               over the training examples as stated in the lecture notes
%               for easy understanding. 
%               However, we recommend you to implement it with vectorization
%               here for efficiency.
%
% Part 7: Implement regularization with the gradients.
%
% Check worksheet for important hints.


%% Part 4 code
%find cost function without regularization


%% Part 5 code
%find cost function with regularization


%% Part 6 code
%partial dervative with vectorization


%% Part 7
%regularized gradient



% =========================================================================

% Retruning unrolled gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end

%% Exercise 4 Neural Network

%  Instructions
%  ------------
% 
%  This file contains code that helps you get started on the
%  neural network exercise. You will need to complete the  
%  following functions in this exericse:
%
%     sigmoidGradient.m
%     randInitializeWeights.m
%     nnCostFunction.m
%     predict.m
%
%  For this exercise, you will not need to change any code in this file,
%  or any other files other than those mentioned above.
%

%% Initialization
clear ; close all; clc

%% Setup the parameters you will use for this exercise
input_layer_size  = 400;  % 20x20 Input Images of Digits
hidden_layer_size = 25;   % 25 hidden units
num_labels = 10;          % 10 labels, from 1 to 10   
                          % (note that we have mapped "0" to label 10)

%% =========== Part 1a: Loading and Visualizing Data =============
%  We start the exercise by first loading and visualizing the dataset. 
%  You will be working with a dataset that contains handwritten digits.
%

% Load Training Data
fprintf('\n1a. Loading and Visualizing Data ...\n')
load('handwritten_digit.mat');
m = size(X, 1);

% Randomly select 100 data points to display
sel = randperm(size(X, 1));
sel = sel(1:100);
fprintf('\nData are randomly selected and displayed\n')
displayData(X(sel, :));

%% ================ Part 1b: Loading Parameters ================
% In this part of the exercise, we load some pre-initialized 
% neural network parameters.
fprintf('\n1b. Loading Saved Neural Network Parameters for testing ...\n')

% Load testing weights into variables Theta1 and Theta2
load('weights_4test.mat');

% Unroll parameters 
nn_params = [Theta1(:) ; Theta2(:)];

fprintf('Program paused. Press enter to continue.\n');
pause;


%% ================ Part 2: Sigmoid Gradient  ================
%  Before you start implementing the neural network, you will first
%  implement the gradient for the sigmoid function. You should complete the
%  code in the sigmoidGradient.m file.
%
% sigmoidGradient is used in nnCostFunction for backpropagation 
% implement: sigmoidGradient.m                           <-------write your code
fprintf('\n2a. Evaluating sigmoid gradient...\n')
g = sigmoidGradient([-1 -0.5 0 0.5 1]);
fprintf('Sigmoid gradient evaluated at [-1 -0.5 0 0.5 1]:\n  ');
fprintf('%f ', g);
fprintf('\nExpect result: \n  0.196612 0.235004 0.250000 0.235004 0.196612\n');
fprintf('Program paused. Press enter to continue.\n');
pause;

% Plot sigmoid gradient function
fprintf('\n2a. Plot sigmoid & sigmoidGradient function\n');

z_plot= linspace(-10,10,50);
g_plot= sigmoid(z_plot);
gprime_plot= sigmoidGradient(z_plot);
figure; hold on;
plot(z_plot, g_plot);
plot(z_plot, gprime_plot);
xlabel('z');
ylabel("g(z) and g'(z)");
legend('sigmoid(z)', 'sigmoidGradient(z)');
hold off;

fprintf('\nProgram paused. Press enter to continue.\n');
pause;

%% ================ Part 3: Initializing Pameters ================
%  In this part of the exercise, you will be starting to implment a two
%  layer neural network that classifies digits. You will start by
%  implementing a function to initialize the weights of the neural network:
%
%  randInitializeWeights.m                            <--------- write your code
%
fprintf('\n3. Initializing Neural Network Parameters ...\n');
% using pi as random seed to fix the random number sequence for testing
% seed can then be removed after testing
initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size, pi);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

fprintf('\nsize(initial_Theta1)=%i x %i', size(initial_Theta1));
fprintf('\npart of initial_Theta1:\n');
fprintf(' %f, ', initial_Theta1(1:5));
fprintf('\nsize(initial_Theta2)=%i x %i', size(initial_Theta2));
fprintf('\npart of initial_Theta2:\n');
fprintf(' %f, ', initial_Theta2(1:5));

fprintf('\n')
% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

fprintf('\nProgram paused. Press enter to continue.\n');
pause;

%% ================ Part 4: Compute Cost (Feedforward) ================
%  To the neural network, you should first start by implementing the
%  feedforward part of the neural network that returns the cost only. You
%  should complete the code in nnCostFunction.m to return cost. After
%  implementing the feedforward to compute the cost, you can verify that
%  your implementation is correct by verifying that you get the same cost
%  as us for the fixed debugging parameters.
%
%  We suggest implementing the feedforward cost *without* regularization
%  first so that it will be easier for you to debug. Later, you
%  will get to implement the regularized cost.
%
%  implement: nnCostFunction.m                        <--------- write your code
%
fprintf('\n4. Feedforward Using Neural Network ...\n')

% No regularization (we set this to 0 here).
lambda = 0;
% nn_params are loaded from weights_4test.mat, 
% X & y are from handwritten_digit.mat
J = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, ...
                   num_labels, X, y, lambda);
fprintf(['\nCost at parameters (lambda = %.2f): %f '...
         '\n(this value should be about 0.287629)\n'], lambda, J);
fprintf('\nProgram paused. Press enter to continue.\n');
pause;

%% =============== Part 5: Implement Regularization Part 1 ===============
%  Once your cost function implementation is correct, you should now
%  continue to implement the regularization with the cost.
%
fprintf('\n5. Checking Cost Function (w/ Regularization) ... \n')

% Weight regularization parameter (we set this to 1 here).
lambda = 3;
J = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, ...
                   num_labels, X, y, lambda);
fprintf(['\nCost at parameters (w/ lambda = %f): %f '...
         '\n(this value should be about 0.576051)\n'], lambda, J);
fprintf('Program paused. Press enter to continue.\n');
pause;


%% =============== Part 6: Implement Backpropagation ===============
%  Once your cost matches up with ours, you should proceed to implement the
%  backpropagation algorithm for the neural network. You should add to the
%  code you've written in nnCostFunction.m to return the partial
%  derivatives of the parameters.
%
checkGradFlag=false %optional, set true to check gradient numerically
%checkGradFlag=true %optional, set true to check gradient numerically
fprintf('\n6. Checking Backpropagation... \n');

if (checkGradFlag==true)
    %  Check gradients by running checkNNGradients
    checkNNGradients; %optional
endif % if (checkGradFlag==true)

% Also output the costFunction debugging values
lambda = 0;
[J  grad]= nnCostFunction(nn_params, input_layer_size, ...
                          hidden_layer_size, num_labels, X, y, lambda);

fprintf(['\nCost after backpropagation implementation \n(lambda = %f): %f ' ...
         '\n(this value should be about 0.287629)\n'], lambda, J);
fprintf('\nPart of the gradient:\n');
fprintf('%f ', grad(1:6));
fprintf('\nExpect result: \n0.000062 0.000094 -0.000193 -0.000168 0.000349 0.000231\n');
fprintf('\nProgram paused. Press enter to continue.\n');
pause;

%% =============== Part 7: Implement Regularization Part 2 ===============
%  Once your backpropagation implementation is correct, you should now
%  continue to implement the regularization with the cost and gradient.
%
fprintf('\n7. Checking Backpropagation (w/ Regularization) ... \n')

%  Check gradients by running checkNNGradients
lambda = 3;
if (checkGradFlag==true)
    checkNNGradients(lambda); %optional
endif % if (checkGradFlag==true)

% Also output the costFunction debugging values
[J grad] = nnCostFunction(nn_params, input_layer_size, ...
                          hidden_layer_size, num_labels, X, y, lambda);

fprintf(['\nCost after backpropagation implementation \n(lambda = %f): %f ' ...
         '\n(this value should be about 0.576051)\n'], lambda, J);
fprintf('\nPart of the gradient:\n');
fprintf('%f ', grad(1:6));
fprintf('\nExpect result: \n0.000062 0.000094 -0.000193 -0.000168 0.000349 0.000231\n');
fprintf('Program paused. Press enter to continue.\n');
pause;


%% =================== Part 8: Training NN ===================
%  You have now implemented all the code necessary to train a neural 
%  network. To train your neural network, we will now use "fmincg", which
%  is a function which works similarly to "fminunc". Recall that these
%  advanced optimizers are able to train our cost functions efficiently as
%  long as we provide them with the gradient computations.
%
fprintf('\n8. Training Neural Network... \n')

%  After you have completed the assignment, change the MaxIter to a larger
%  value to see how more training helps.
options = optimset('MaxIter', 50);
%options = optimset('MaxIter', 10);

%  You should also try different values of lambda
lambda = 1;

% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X, y, lambda);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

save("weights_trained_by_me.mat", "Theta1", "Theta2")
fprintf('weights_trained_by_me.mat saved. Press enter to continue.\n');
pause;


%% ================= Part 9: Visualize Weights =================
%  You can now "visualize" what the neural network is learning by 
%  displaying the hidden units to see what features they are capturing in 
%  the data.
fprintf('\n9. Visualizing Neural Network... \n')
figure;
displayData(Theta1(:, 2:end));

fprintf('\nProgram paused. Press enter to continue.\n');
pause;

%% ================= Part 10: Implement Predict =================
%  After training the neural network, we would like to use it to predict
%  the labels. You will now implement the "predict" function to use the
%  neural network to predict the labels of the training set. This lets
%  you compute the training set accuracy.
%
%  implement: predict.m                               <--------- write your code
pred = predict(Theta1, Theta2, X);

fprintf('\n10. Training Set Accuracy: %f\n', mean(double(pred == y)) * 100);

fprintf('\nProgram paused. Press enter to continue.\n');
pause;

%% ================= Part 11: Manual Evaluation =================
%  To give you an idea of the network's output, you can also run
%  through the examples one at the a time to see what it is predicting.
fprintf('\n11. Manual Evaluation . . .\n');
%  Randomly permute examples
rp = randperm(m);

figure;
for i = 1:m
    % Display 
    fprintf('\nDisplaying Example Image');
    displayData(X(rp(i), :));

    pred = predict(Theta1, Theta2, X(rp(i),:));
    % mod(a, 10), divide a by 10 and take remainder, mod(10,10) give 0
    fprintf('\nNeural Network Prediction: %d (expect: %d)\n', mod(pred, 10), mod(y(rp(i)),10));
    
    % Pause with quit option
    s = input('Paused - press enter to continue, q to exit:','s');
    if s == 'q'
      break
    end
end



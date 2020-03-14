function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
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
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% converting y to binary vector
y_ = zeros(m,num_labels);
y_(sub2ind(size(y_),1:numel(y),y')) = 1;
y_ = y_';   % y_ each col is vector of an data, 10x5000

% feedforward for 3 layers
a1 = X';

a1 = [ones(1,size(a1,2)); a1];
z2 = Theta1*(a1);
a2 = sigmoid(z2);

a2 = [ones(1,size(a2,2)); a2];
z3 = Theta2*(a2);
a3 = sigmoid(z3);

h = a3; % 10x5000

% computing cost
one_ = ones(num_labels,m);
J = -(sum(sum(y_.*log(h) + (one_-y_).*log(one_-h), 1), 2))/m;
J = J + lambda*( sum((Theta1(:,2:end).^2),'all') + sum((Theta2(:,2:end).^2),'all')) / (2*m);

% backpropogation
    
    del3 = a3 - y_; % 10x5000
    temp = (Theta2)'*del3;
    temp2 = [ones(1,size(z2,2));; sigmoidGradient(z2)];
    del2 = temp.*temp2; % 26x5000
    del2 = del2(2:end,:);
    
    Theta2_acc = zeros(size(Theta2_grad));
    Theta1_acc = zeros(size(Theta1_grad));
    for i=1:m
        Theta2_acc = Theta2_acc + del3(:,i)*(a2(:,i)');
        Theta1_acc = Theta1_acc + del2(:,i)*(a1(:,i)');
    end

% alternate entire loop implementation
%     Theta2_acc = zeros(size(Theta2_grad));
%     Theta1_acc = zeros(size(Theta1_grad));
%     for i=1:m
%         a1 = X(i,:);
%         
%         a1 = a1';
%         a1 = [ones(1,size(a1,2)); a1];
%         z2 = Theta1*(a1);
%         a2 = sigmoid(z2);
%         a2 = [ones(1,size(a2,2)); a2];
%         z3 = Theta2*(a2);
%         a3 = sigmoid(z3);
%         
%         del3 = a3 - y_(:,i);
%         
%         del2 = (Theta2)'*del3.*[1;sigmoidGradient(z2)];
%         del2 = del2(2:end,:);
%         
%         Theta2_acc = Theta2_acc + del3*(a2');
%         Theta1_acc = Theta1_acc + del2*(a1');
%         
%     end
    
    Theta2_grad = Theta2_acc/m;
    Theta1_grad = Theta1_acc/m;

    Theta1_reg = (lambda/m)*Theta1;
    Theta1_reg(:,1) = 0;
    Theta2_reg = (lambda/m)*Theta2;
    Theta2_reg(:,1) = 0;
    
    Theta1_grad = Theta1_grad + Theta1_reg;
    Theta2_grad = Theta2_grad + Theta2_reg;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end

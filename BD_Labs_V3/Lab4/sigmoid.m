function g = sigmoid(z)
%sigmoid Compute sigmoid functoon
%   J = sigmoid(z) computes the sigmoid of z.

g = 1.0 ./ (1.0 + exp(-z));
end

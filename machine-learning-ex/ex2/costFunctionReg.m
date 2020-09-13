function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


hx = sigmoid(X * theta);

J = (sum(-y' * log(hx) - (1 - y')*log(1 - hx)) / m) + lambda * sum(theta(2:end).^2) / (2*m);

grad =((hx - y)' * X / m)' + lambda .* theta .* [0; ones(length(theta)-1, 1)] ./ m ;

%predictions = sigmoid(X * theta);

%p1 = ((y') * log(predictions));
%p2 = ((1- y') * (log(1 - predictions)));

%initJ = ((-1/m) * (p1 + p2));
%theta(1) = 0;
%regularized = ((lambda/2 * m) * sum((theta(2:end) .^ 2) /2*m));

%J = initJ + regularized;

%grad(1) = (1/m) * ((predictions - y)' * X(:,1));

%grad(2) = ((1/m) * ((predictions - y)' * X(:,2))) + ((lambda/m) * theta(2));
%grad(3) = ((1/m) * ((predictions - y)' * X(:,3))) + ((lambda/m) * theta(3));


% =============================================================

end

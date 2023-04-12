function [coefficients, std_errors, t_stats, r_squared, adj_r_squared, durbin_watson] = fOls(y, X)

% Input: X - Independent variable(s) matrix (nxp), where n is the number of observations, and p is the number of independent variables.
%        y - Dependent variable vector (nx1)

% Output: coefficients - OLS estimated coefficients (px1)
%         std_errors - Standard errors of the estimated coefficients (px1)
%         t_stats - t-statistics of the estimated coefficients (px1)
%         r_squared - Coefficient of determination (R-squared)
%         adj_r_squared - Adjusted R-squared
%         durbin_watson - Durbin-Watson statistic

% Number of observations
n = size(X, 1);

% Add a column of ones to X for the constant term
% X = [ones(n, 1), X];

% Estimate the OLS coefficients
coefficients = (X' * X) \ (X' * y);

% Compute the residuals
residuals = y - X * coefficients;

% Calculate the residual sum of squares
RSS = residuals' * residuals;

% Calculate the total sum of squares
TSS = (y' * y) - (sum(y)^2) / n;

% Compute the R-squared
r_squared = 1 - (RSS / TSS);

% Compute the adjusted R-squared
adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - size(X, 2) - 1);

% Estimate the variance of the error term
sigma2 = RSS / (n - size(X, 2));

% Compute the variance-covariance matrix
var_covar = sigma2 * inv(X' * X);

% Compute the standard errors
std_errors = sqrt(diag(var_covar));

% Calculate t-statistics
t_stats = coefficients ./ std_errors;

% Calculate the Durbin-Watson statistic
durbin_watson = sum(diff(residuals).^2) / sum(residuals.^2);

end

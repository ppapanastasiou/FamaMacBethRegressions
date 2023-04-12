% Removes all items in Environment!
clear all;

% Clear the console
clc;

% Read CSV
data = readtable("FamaFrench.csv");

% Split using both named columns and numeric indexing for larger blocks
dates = data.date;
factors = data(:, {'VWMe', 'SMB', 'HML'});
riskfree = data.RF;
portfolios = data(:, 6:size(data,2));

% Use matrix for easier linear algebra
factors = table2array(factors);
%riskfree = table2array(riskfree);
portfolios = table2array(portfolios);

% Shape information
t = size(factors, 1);
k = size(factors, 2);
n = size(portfolios, 2);

% Replicate riskfree and compute excess returns

% repmat the risk free vector n times
riskfree_replicated =repmat(riskfree, n, 1);
riskfree_replicated = reshape(riskfree_replicated, n, t)'; % Transpose the result with '

% % Loop version
% % repmat the risk free vector n times
% riskfree_repmat =repmat(riskfree, n, 1);
% 
% %Initialize the matrix
% riskfree_replicated = zeros(t, n);
% 
% % Fill the matrix row-wise using a loop
% counter = 1;
% for r = 1:t
%     for c = 1:n
%         riskfree_replicated(r, c) = riskfree_repmat(counter);
%         counter = counter + 1;
%     end
% end

% riskfree_replicated = repmat(riskfree, 1, n);
excess_returns = portfolios - riskfree_replicated;

% Time series regressions
x = [ones(t, 1), factors];
% ts_res = fitlm(x, excess_returns, 'Intercept', false);
mTsCoeff = x\excess_returns;
alpha = mTsCoeff(1, :);
beta = mTsCoeff(2:end, :);

% Calculate average excess returns
avgexcess_returns = mean(excess_returns);

% Cross-section regression
% cs_res = fitlm(beta_transposed', avgexcess_returns', 'Intercept', false);
% [b,bse,res,n,rss,r2] = OLSest(avgexcess_returns',beta',1);
% [coefficients, std_errors, t_stats, r_squared, adj_r_squared, durbin_watson] = fOls(avgexcess_returns', beta')
mCsCoeff = beta'\avgexcess_returns';
risk_premia = mCsCoeff;

% Moment conditions
X = [ones(t, 1), factors];
p = [alpha; beta];
epsilon = excess_returns - X * p;
moments1 = kron(epsilon, ones(1, k + 1));
moments1 = moments1 .* kron(ones(1, n), X);

% Calculate u matrix
risk_premia_beta = risk_premia' * beta;
risk_premia_beta = repmat(risk_premia_beta, t, 1);
u = excess_returns - risk_premia_beta;

moments2 = u * beta';

% Score covariance
S = cov([moments1, moments2]);
t_value = size(X, 1);

% Jacobian
G = zeros(n * k + n + k, n * k + n + k);
sigma_x = (X' * X) / t_value;
G(1:(n * k + n), 1:(n * k + n)) = kron(eye(n), sigma_x);
G((n * k + n + 1):end, (n * k + n + 1):end) = -beta * beta';

for i = 1:n
    temp = zeros(k, k + 1);
    values = mean(u(:, i)) - beta(:, i) .* risk_premia;
    temp(sub2ind(size(temp), 1:k, 2:k + 1)) = values;
    G((n * k + n + 1):end, ((i - 1) * (k + 1) + 1):(i * (k + 1))) = temp;
end

% vcv = (inv(G') * S * inv(G)) / t_value;
vcv = (G'\S /G) / t_value;

% Calculate vcv_alpha
vcv_alpha = vcv(1:4:(n * k + n), 1:4:(n * k + n));

% Calculate J
J = alpha * inv(vcv_alpha) * alpha';
J = J(1, 1);

% Calculate Jpval
Jpval = 1 - chi2cdf(J, n);

% Extract vcvrisk_premia
vcvrisk_premia = vcv((n * k + n + 1):end, (n * k + n + 1):end);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Extract vcvrisk_premia
vcvrisk_premia = vcv((n * k + n + 1):end, (n * k + n + 1):end);

% Annualized Risk Premia
annualized_rp = 12 * risk_premia;
arp = annualized_rp(:);
arp_se = sqrt(12 * diag(vcvrisk_premia));

fprintf('        Annualized Risk Premia\n');
fprintf('           Market       SMB        HML\n');
fprintf('--------------------------------------\n');
fprintf('Premia     %.4f    %.4f     %.4f\n', arp(1), arp(2), arp(3));
fprintf('Std. Err.  %.4f    %.4f     %.4f\n', arp_se(1), arp_se(2), arp_se(3));
fprintf('\n\n');

fprintf('J-test:   %.4f\n', J);
fprintf('P-value:   %.4f\n', Jpval);

i = 0;
beta_se = [];
for j = 1:5
    for m = 1:5
        a = alpha(i + 1);
        b = beta(:, i + 1);
        variances = diag(vcv(((k + 1) * i + 1):((k + 1) * (i + 1)), ((k + 1) * i + 1):((k + 1) * (i + 1))));
        beta_se = [beta_se; sqrt(variances)];
        s = sqrt(variances);
        c = [a; b];
        t = c ./ s;
        fprintf('Size: %d, Value: %d   Alpha   Beta(VWM)   Beta(SMB)   Beta(HML)\n', j, m);
        fprintf('Coefficients: %10.4f  %10.4f  %10.4f  %10.4f\n', a, b(1), b(2), b(3));
        fprintf('Std Err.      %10.4f  %10.4f  %10.4f  %10.4f\n', s(1), s(2), s(3), s(4));
        fprintf('T-stat        %10.4f  %10.4f  %10.4f  %10.4f\n', t(1), t(2), t(3), t(4));
        fprintf('\n');
        i = i + 1;
    end
end

% Convert beta_se to an array
beta_se = reshape(beta_se, [], 1);

final_fama_macbeth_results = struct('alpha', alpha, 'beta', beta, 'beta_se', beta_se, 'arp_se', arp_se, 'arp', arp, 'J', J, 'Jpval', Jpval);




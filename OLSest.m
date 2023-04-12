function [b,bse,res,n,rss,r2] = OLSest(y,x,output)
% This function performs an OLS estimation
% function [b,bse,res,n,rss,r2] = OLSest(y,x,output)
% input:    y, vector with dependent variable
%           x, matrix with explanatory variable 
%               function will automatically add a constant if the first col
%               is not a vector  of ones
%           output, 1 = printed output
% output:   b, estimated parameters
%           bse, standard errors for bhat
%           res, estimated residuals
%           n, number of observations used
%           rss, residual sum of squares
%           r2, Rsquared

% select those rows that have observations for all variables
ninit = length(y);
testnan = [isnan(y) isnan(x)];
testnan = (sum(testnan,2)==0);
y = y(testnan);
x = x(testnan,:);
% % test whether first column is vector of ones
% temp = (x(x(:,1)==1));
% if length(temp) ~= length(x)
%   x = [ones(length(x),1) x];  % add constant if not included in x
% end



[n,k] = size(x);        % sample size - n, number of explan vars (incl constant) - k   
xxi   = inv(x'*x);      % Note that this is the inefficient way of calculating 
                        % the inverse of x'*x, but as xxi is required later for 
                        % the calculation of bse, we are not really loosing
                        % anything
b     = xxi*x'*y;
res   = y - x*b;
rss   = res'*res;
ssq   = rss/(n-k);
s     = sqrt(ssq);
bse   = ssq*xxi;
bse   = sqrt(diag(bse));
tstat = b./bse;
ym    = y - mean(y);
r2    = 1 - (res'*res)/(ym'*ym);
adjr2 = 1 - (n-1)*(1-r2)/(n-k);
fstat = ((((ym'*ym))-(res'*res))/(k-1))/((res'*res)/(n-k));
dw    = corrcoef([res(1:end-1) res(2:end)]);
dw    = 2*(1-dw(2,1)); 
 
if output

    % calculate p values (requires either MATLAB stats toolbox or NAG toolbox

try      % if stats toolbox is available
    pval  = 2*(1-tcdf(abs(tstat),n-k));
    pvalf = 1- fcdf(fstat,k-1,n-k);
catch
    try     % if NAG toolbox is available
        pval  = 2*(1-g01eb(abs(tstat),n-k));
        pvalf = g01ed(fstat,k-1,n-k,'tail','U');
    catch
        pval = -999*ones(size(tstat));
        pvalf = -999;
    end
end

fprintf('===========================================================\n');
fprintf('===== Regression Output  ==================================\n');
fprintf('Obs used = %4.0f, missing obs = %4.0f \n',n,(ninit-n));
fprintf('Rsquared = %5.4f \n',r2);
fprintf('adj_Rsq  = %5.4f \n',adjr2);
fprintf('===== Estimated Model Parameters ==========================\n');
fprintf('=   Par       se(Par)   t(Par)    pval  ==================\n');
format short;
disp([b bse tstat pval]);
fprintf('===== Model Statistics ====================================\n');
fprintf(' Fstat = %5.4f (%5.4f)\n',[fstat;pvalf]);
fprintf(' standard error = %5.4f\n',sqrt(ssq));
fprintf(' RSS = %5.4f\n',rss);
fprintf(' Durbin-Watson  = %5.4f\n',dw);
fprintf('===========================================================\n');
fprintf('== p-values of -999 indicate that neither the stat ========\n'); 
fprintf('== nor the NAG toolbox were available =====================\n');

end
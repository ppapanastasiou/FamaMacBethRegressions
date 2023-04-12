#Removes all items in Environment!
rm(list=ls()) 
# Clear the console
cat("\014")

# Read CSV
data <- read.csv("FamaFrench.csv")

# Split using both named columns and numeric indexing for larger blocks
dates <- data[["date"]]
factors <- data[, c("VWMe", "SMB", "HML")]
riskfree <- data[["RF"]]
portfolios <- data[, 6:ncol(data)]

# Use matrix for easier linear algebra
factors <- as.matrix(factors)
riskfree <- as.matrix(riskfree)
portfolios <- as.matrix(portfolios)

# Shape information
t <- nrow(factors)
k <- ncol(factors)
n <- ncol(portfolios)

# Replicate riskfree and compute excess returns
riskfree_replicated <- matrix(rep(riskfree, n), nrow = t, byrow = TRUE)
excess_returns <- portfolios - riskfree_replicated


# Load required libraries
library(stats)
library(plm)

# Time series regressions
x <- cbind(1, factors)
ts_res <- lm(excess_returns ~ x - 1) # '-1' to remove the intercept term, as it is included in 'x'
alpha <- coef(ts_res)[1,]
beta <- coef(ts_res)[-1,]

# Calculate average excess returns
avgexcess_returns <- colMeans(excess_returns)

# Transpose beta matrix
beta_transposed <- t(beta)

# Cross-section regression
cs_res <- lm(avgexcess_returns ~ beta_transposed - 1) # '-1' to remove the intercept term 
# (cross sectional regression without intercept). Remember that it can be run with and without intercept.
risk_premia <- coef(cs_res)


# Load required libraries
library(Matrix)

# Moment conditions
X <- cbind(1, factors)
p <- rbind(alpha, beta)
epsilon <- excess_returns - X %*% p
moments1 <- kronecker(epsilon, matrix(1, nrow = 1, ncol = k + 1))
moments1 <- moments1 * kronecker(matrix(1, nrow = 1, ncol = n), X)

# Calculate u matrix
risk_premia_beta <- risk_premia %*% beta
risk_premia_beta <- matrix(rep(risk_premia_beta, each = nrow(excess_returns)), nrow = nrow(excess_returns), byrow = TRUE)
u <- excess_returns - risk_premia_beta

moments2 <- u %*% t(beta)

# Score covariance
S <- as.matrix(cov(cbind(moments1, moments2)))
t_value <- nrow(X)

# Jacobian
G <- matrix(0, nrow = n * k + n + k, ncol = n * k + n + k)
sigma_x <- (t(X) %*% X) / t_value
G[1:(n * k + n), 1:(n * k + n)] <- kronecker(diag(n), sigma_x)
G[(n * k + n + 1):ncol(G), (n * k + n + 1):ncol(G)] <- -beta %*% t(beta)

for (i in 1:n) {
  temp <- matrix(0, nrow = k, ncol = k + 1)
  values <- mean(u[, i]) - beta[, i] * risk_premia
  diag(temp[, -1]) <- values
  G[(n * k + n + 1):ncol(G), ((i - 1) * (k + 1) + 1):(i * (k + 1))] <- temp
}

vcv <- solve(t(G)) %*% S %*% solve(G) / t_value

################################################################################

# Calculate vcv_alpha
vcv_alpha <- vcv[seq(1, n * k + n, by = 4), seq(1, n * k + n, by = 4)]

# Calculate J
J <- t(alpha) %*% solve(vcv_alpha) %*% alpha
J <- J[1, 1]

# Calculate Jpval
Jpval <- 1 - pchisq(J, df = 25)




################################################################################
# Extract vcvrisk_premia
vcvrisk_premia <- vcv[(n * k + n + 1):nrow(vcv), (n * k + n + 1):ncol(vcv)]

# Annualized Risk Premia
annualized_rp <- 12 * risk_premia
arp <- as.vector(annualized_rp)
arp_se <- sqrt(12 * diag(vcvrisk_premia))

cat("        Annualized Risk Premia\n")
cat("           Market       SMB        HML\n")
cat("--------------------------------------\n")
cat(sprintf("Premia     %0.4f    %0.4f     %0.4f\n", arp[1], arp[2], arp[3]))
cat(sprintf("Std. Err.  %0.4f    %0.4f     %0.4f\n", arp_se[1], arp_se[2], arp_se[3]))
cat("\n\n")

cat(sprintf("J-test:   %0.4f\n", J))
cat(sprintf("P-value:   %0.4f\n", Jpval))

i <- 0
beta_se <- vector()
for (j in 1:5) {
  for (m in 1:5) {
    a <- alpha[i + 1]
    b <- beta[, i + 1]
    variances <- diag(vcv[((k + 1) * i + 1):((k + 1) * (i + 1)), ((k + 1) * i + 1):((k + 1) * (i + 1))])
    beta_se <- c(beta_se, sqrt(variances))
    s <- sqrt(variances)
    c <- c(a, b)
    t <- c / s
    cat(sprintf("Size: %d, Value: %d   Alpha   Beta(VWM)   Beta(SMB)   Beta(HML)\n", j, m))
    cat(sprintf("Coefficients: %10.4f  %10.4f  %10.4f  %10.4f\n", a, b[1], b[2], b[3]))
    cat(sprintf("Std Err.      %10.4f  %10.4f  %10.4f  %10.4f\n", s[1], s[2], s[3], s[4]))
    cat(sprintf("T-stat        %10.4f  %10.4f  %10.4f  %10.4f\n", t[1], t[2], t[3], t[4]))
    cat("\n")
    i <- i + 1
  }
}


# Convert beta_se to an array
beta_se <- array(beta_se)

final_fama_macbeth_results<- list(alpha = alpha, beta = beta, beta_se = beta_se, arp_se = arp_se, arp = arp, J = J, Jpval = Jpval)

# Save the results to an RDS file
# saveRDS(final_fama_macbeth_results,
#        file = "fama-macbeth-results.rds")



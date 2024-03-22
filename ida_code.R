#Question 2

library(mice)
# Load the data
load("dataex2.Rdata")

# Define the function to calculate coverage for β1
calculate_coverage <- function(data, method) {
  coverages <- numeric(length = 100) # Initialize vector for coverage checks
  
  for (i in 1:100) {
    set.seed(1 + i) # For reproducibility, vary seed with iteration
    imp <- mice(data[, , i], method = method, m = 20, maxit = 5, seed = 1)
    fit <- with(imp, lm(Y ~ X))
    poolFit <- pool(fit)
    summaryFit <- summary(poolFit)
    
    # Calculate 95% CI bounds for β1
    estimate <- summaryFit$estimate[2]  # β1 estimate for X
    std_error <- summaryFit$std.error[2]  # Standard error for β1
    ci_lower <- estimate - 1.96 * std_error
    ci_upper <- estimate + 1.96 * std_error
    
    # Check if the true β1 (3) is within the CI
    coverages[i] <- ci_lower <= 3 && ci_upper >= 3
  }
  
  # Calculate empirical coverage probability
  mean(coverages)
}

# Calculate coverage for both methods
coverage_stochastic <- calculate_coverage(dataex2, method = "norm.nob")
coverage_bootstrap <- calculate_coverage(dataex2, method = "norm.boot")

# Output the coverage results
cat("Coverage for Stochastic Regression Imputation:", coverage_stochastic, "\n")
cat("Coverage for Bootstrap Based Imputation:", coverage_bootstrap, "\n")


#Question 3
# Load necessary library
library(maxLik)

# Load the data
load("dataex3.Rdata")

# Define the log-likelihood function for a given mu
calculateLogLikelihood <- function(mu) {
  dataPoints <- dataex3$X
  responseIndicator <- dataex3$R
  variance <- 1.5^2  # Defined as sigma squared
  
  # Calculate log-likelihood
  logLikelihood <- sum(responseIndicator * dnorm(dataPoints, mean = mu, sd = sqrt(variance), log = TRUE) +
                         (1 - responseIndicator) * pnorm(dataPoints, mean = mu, sd = sqrt(variance), log.p = TRUE))
  
  return(logLikelihood)
}

# Optimize to find the MLE of mu using the BFGS method
optimizationResult <- optim(par = 0, fn = function(mu) -calculateLogLikelihood(mu), method = "BFGS")

# Extract the MLE of mu from the optimization results
muMLE <- optimizationResult$par

# Output the MLE of mu
cat("Maximum Likelihood Estimate (MLE) of mu:", muMLE, "\n")


# Question 5
library(maxLik)

# Load data and prepare
load("dataex5.Rdata")
dataset <- dataex5

# Log-likelihood function for logistic regression
calcLogLikelihood <- function(params, predictors, outcomes) {
  predicted_probs <- 1 / (1 + exp(-(params[1] + params[2] * predictors)))
  log_likelihood <- sum(log(predicted_probs[outcomes == 1])) + sum(log(1 - predicted_probs[outcomes == 0]))
  return(log_likelihood)
}

# M-step using the maxLik package
performMStep <- function(logLikFunc, initialParams, predictors, outcomes) {
  maxLik(logLikFunc, start = initialParams, predictors = predictors, outcomes = outcomes)
}

# Initialize parameters
initial_params <- c(0, 0)

# Define missing data
missing_data <- is.na(dataset$Y)

# EM algorithm setup
convergence_threshold <- 1e-8
max_iterations <- 1000
iterations <- 0
params <- initial_params
has_converged <- FALSE

while (!has_converged && iterations < max_iterations) {
  iterations <- iterations + 1
  
  # E-step: Estimate probabilities for missing Y
  missing_probs <- 1 / (1 + exp(-(params[1] + params[2] * dataset$X[missing_data])))
  
  # Replace missing Y values with estimated probabilities
  outcomes_complete <- dataset$Y
  outcomes_complete[missing_data] <- missing_probs
  
  # M-step: Update logistic regression model with complete data
  m_step_result <- performMStep(calcLogLikelihood, params, dataset$X, outcomes_complete)
  updated_params <- coef(m_step_result)
  
  # Check for convergence
  if (max(abs(updated_params - params)) < convergence_threshold) {
    has_converged <- TRUE
  } else {
    params <- updated_params
  }
}

# Final parameter estimates
final_params <- params
cat("Final parameter estimates: ", final_params)

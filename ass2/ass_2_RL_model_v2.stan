//
// This Stan program defines a simple model, with a
// vector of values 'y' modeled as normally distributed
// with mean 'mu' and standard deviation 'sigma'.
//
// Learn more about model development with Stan at:
//
//    http://mc-stan.org/users/interfaces/rstan.html
//    https://github.com/stan-dev/rstan/wiki/RStan-Getting-Started
//
// The input data is 
data {
  int<lower=1, upper=1000> n; // Number of trials with lower and upper bound
  array[n] int choice; // this is an array (matrix), with a binary integer choice
  array[n] int feedback; // this is an array (matrix), with the feedback
}

transformed data {
  array[n] int othersChoice; // We calculate this from Choice and Feedback
  
  // Below we generate the others choices for each trial (used for PE)
  for (t in 1:n){
    if (feedback[t] == 1) // meaning you win
      othersChoice[t] = choice[t]; // you had the same choices (you guessed the hand)
    else if (feedback[t] == 0) // you lost, mismatch
	 othersChoice[t] = 1 - choice[t]; // not guessed the hand
  }
}

// The parameters accepted by the model
parameters {
  real <lower = 0, upper =1> learningRate; // Defining lower and upper bound for learning rate
}



// The model to be estimated
model {

  // Define prior of learningRate below
  target += beta_lpdf(learningRate | 1, 2); // indicating that smaller learning rates are more likely

  // likelihood function here
  real EV; // Expected Value
  real PE; // Prediction Error

  // The first trial is coded below
  EV = 0.5;

  for(t in 1:n){
    
    target += bernoulli_lpmf(choice[t] | EV); // add log probability of making choice[t] given rate EV
    
    PE = othersChoice[t] - EV; // calculate prediction error for this trial
    EV = EV + learningRate * PE; // the EV on Right Hand Side (RHS) is the same as prevEV

  }
}

generated quantities {
  real learningRate_prior;
  real PE;
  real EV;
  array[n] real prior_preds;
  array[n] real post_preds;
  
  
  learningRate_prior = beta_rng(1,2); // generating the prior distribution
  
  EV = 0.5;
  
  // Generate posterior predictions
  for (t in 1:n){
    post_preds[t] = EV; // We save the EV samples as the posterior predictions
    
    PE = othersChoice[t] - EV;
    EV = EV + learningRate * PE; // Compute EV samples using the posterior distribution of learningRate
  }
  
  // Reset EV
  EV = 0.5;
  
  // Generate prior predictions
  for (t in 1:n){
    prior_preds[t] = EV; // Same as above
    
    PE = othersChoice[t] - EV;
    EV = EV + learningRate_prior * PE; // Compute EV samples using the prior distribution of learningRate
  }
}



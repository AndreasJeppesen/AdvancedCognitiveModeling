
// This model implements the Simple Bayes cognitive model


// We define a normal random number generator function, that has a lower bound. Thank you Riccardo :)
functions{
  real normal_lb_rng(real mu, real sigma, real lb) { // normal distribution with a lower bound
    real p = normal_cdf(lb | mu, sigma);  // cdf for bounds
    real u = uniform_rng(p, 1);
    return (sigma * inv_Phi(u)) + mu;  // inverse cdf for value
  }
}

// The input data is a vector 'y' of length 'N'.
data {
  int n; // number of trials
  array[n] int<lower=1, upper = 8> FirstRating;
  array[n] int<lower=1, upper = 8> GroupRating;
  array[n] int<lower=1, upper = 8> SecondRating;
}

transformed data {
  array[n] real p_FirstRating;
  array[n] real p_GroupRating;
  array[n] real p_SecondRating;
  
  array[n] real L_FirstRating;
  array[n] real L_GroupRating;
  array[n] real L_SecondRating;
  
  for (i in 1:n){
    p_FirstRating[i] = FirstRating[i] / 9.0;  // turning ratings to ]0,1[ space.
    p_SecondRating[i] = SecondRating[i] / 9.0;
    p_GroupRating[i] = GroupRating[i] / 9.0;  // NB: "/ 9" is integer division, rounding to 0. Use "/ 9.0" !!
    
    L_FirstRating[i] = logit(p_FirstRating[i]); // turning into log-odds space
	  L_SecondRating[i] = logit(p_SecondRating[i]);
	  L_GroupRating[i] = logit(p_GroupRating[i]);
    
  }

  
}

// The parameters accepted by the model. Our model
// accepts two parameters 'mu' and 'sigma'.
parameters {
  real bias;
  real<lower=0> error; // Bounding error to be positive
}

// The model to be estimated.
model {
  
  // We are using a zero centered normal prior for the bias:
  target += normal_lpdf(bias | 0, 2);
  
  // The prior for error is a truncated normal (only positive values).
  // Note: we subtract the cumulative density of sampling 0 or higher, ie. we remove the negative half of the distribution
  target += normal_lpdf(error | 0, 2) - normal_lccdf(0 | 0, 2);
  
  // The likelihood function:
  target += normal_lpdf(to_vector(L_SecondRating) | bias + to_vector(L_FirstRating) + to_vector(L_GroupRating), error);
}

generated quantities {
  real bias_prior;
  real error_prior;
  array[n] real log_lik;
  
  
  bias_prior = normal_rng(0,2); // generating the bias prior
  error_prior = normal_lb_rng(0,2,0);  // generating the error prior using the function we defined earlier
  
  // Saving the log_likelihood. Same as the likelihood function in the model chunk.
  for (t in 1:n){
    log_lik[t] = normal_lpdf(L_SecondRating[t] | bias + L_FirstRating[t] + L_GroupRating[t], error);
  }
  
}


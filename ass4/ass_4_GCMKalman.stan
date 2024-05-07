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


// The input data
data {
  int<lower=1> ntrials; // number of trial
  int<lower=1> nfeats; // number of features
  array[ntrials, nfeats] int<lower=0, upper=1> stimuli; // observed features
  //array[ntrials] int<lower=1, upper=4> category; // true category
  array[ntrials] int<lower=1, upper=4> response; // responded category
  array[ntrials] int<lower=0, upper=1> dangerous; // observed danger
  array[ntrials] int<lower=0, upper=1> nutricious; // observed nutrition
  vector[nfeats] w_prior_values; // The priors for w of each feature
}

transformed data{
  array[ntrials] int resp_d; // dividing response into danger and nutrition
  array[ntrials] int resp_n;
  
  for (t in 1:ntrials){
    if (response[t] == 1) {
      resp_d[t] = 0;
      resp_n[t] = 0;
    }
    if (response[t] == 2) {
      resp_d[t] = 0;
      resp_n[t] = 1;
    }
    if (response[t] == 3) {
      resp_d[t] = 1;
      resp_n[t] = 0;
    }
    if (response[t] == 4) {
      resp_d[t] = 1;
      resp_n[t] = 1;
    }
  }
  
  // flagging when we observed the different types of aliens
  array[ntrials] int dang1_obs;
  array[ntrials] int dang0_obs;
  array[ntrials] int nutr1_obs;
  array[ntrials] int nutr0_obs;
  
  for (t in 1:ntrials){
    if (dangerous[t] == 1){
      dang1_obs[t] = 1;
      dang0_obs[t] = 0;
    }
    else if (dangerous[t] == 0){
      dang1_obs[t] = 0;
      dang0_obs[t] = 1;
    }
    if (nutricious[t] == 1){
      nutr1_obs[t] = 1;
      nutr0_obs[t] = 0;
    }
    else if (nutricious[t] == 0){
      nutr1_obs[t] = 0;
      nutr0_obs[t] = 1;
    }
  }
  
}


// The parameters accepted by the model
parameters {
  simplex[nfeats] w; // weight for every feature, summing to 1
  // real kappa; // concentration parameter for the distribution of weights. Only needed for multilevel?
  real logit_c;  // we sample c on the logit scale, as it is bounded up- and downwards
  real log_R;    // we sample R on log scale, as it is bounded downwards by zero
}

transformed parameters{
  // Transformation of parameters to etsimate
  real c;
  real R;
  c = inv_logit(logit_c) * 4 ; // go from sampled c to bounded c
  R = exp(log_R); // go from sampled R to bounded R
  //vector[nfeats] alpha = kappa * w; // only needed for multilevel?
  
  
  // Applying calculations of distance, similarity and Kalman update given the stimuli
  array[ntrials+1, nfeats] real dang1_pos;
  array[ntrials+1, nfeats] real dang0_pos;
  array[ntrials+1, nfeats] real nutr1_pos; // positions in 5D space
  array[ntrials+1, nfeats] real nutr0_pos;
  
  array[ntrials+1] real dang1_K;
  array[ntrials+1] real dang0_K;
  array[ntrials+1] real nutr1_K; // Kalman gains
  array[ntrials+1] real nutr0_K;
  
  array[ntrials+1] real dang1_P;
  array[ntrials+1] real dang0_P;
  array[ntrials+1] real nutr1_P; // P (used in Kalman update)
  array[ntrials+1] real nutr0_P;
  
  array[ntrials] real d_rate;
  array[ntrials] real n_rate; // rates of responding danger and nutrition - pr trial
  
  
  // Set initial values for positions
  for (f in 1:nfeats){
    dang1_pos[1,f] = 0.5;
    dang0_pos[1,f] = 0.5;
    nutr1_pos[1,f] = 0.5;
    nutr0_pos[1,f] = 0.5;
  }
  
  // set initial values for P
  dang1_P[1] = 1;
  dang0_P[1] = 1;
  nutr1_P[1] = 1;
  nutr0_P[1] = 1;
  
  // loop over trials to evaluate stimuli
  for (t in 1:ntrials){
    
    vector[nfeats] alien; // save observed alien as vector
    for (f in 1:nfeats){
      alien[f] = stimuli[t, f];
    }
    
    if (t == 1){
      d_rate[t] = 0.5; // in first trial, response is given randomly
      n_rate[t] = 0.5;
    }
    else if (t >= 1){
      
      if (sum(dang1_obs[:(t-1)]) == 0 || sum(dang0_obs[:(t-1)]) == 0){ // if we havent observed both d1 and d0
        d_rate[t] = 0.5;
      } else{
        // distance and similarity stuff
        real dang1_dist;
        real dang0_dist;
        
        array[nfeats] real fwise_dist_d1; // feature-wise distance
        array[nfeats] real fwise_dist_d0;
        
        real dang1_sim; // similarities
        real dang0_sim;
        
        for (f in 1:nfeats){
          fwise_dist_d1[f] = w[f] * (alien[f] - dang1_pos[t,f])^2; // square the distances one by one
          fwise_dist_d0[f] = w[f] * (alien[f] - dang0_pos[t,f])^2;
        }
        dang1_dist = sqrt(sum(fwise_dist_d1));
        dang0_dist = sqrt(sum(fwise_dist_d0)); // sum them and take sqrt
        
        dang1_sim = exp(-c * dang1_dist);
        dang0_sim = exp(-c * dang0_dist); // calc similarity
        
        d_rate[t] = dang1_sim / (dang1_sim + dang0_sim); // rate calculated Luce's axiom
        
        
      }
      
      if (sum(nutr1_obs[:(t-1)]) == 0 || sum(nutr0_obs[:(t-1)]) == 0){ // if we havent observed both n1 and n0
        n_rate[t] = 0.5;
      } else{
        // distance and similarity stuff
        real nutr1_dist;
        real nutr0_dist;
        
        array[nfeats] real fwise_dist_n1; // feature-wise distance
        array[nfeats] real fwise_dist_n0;
        
        real nutr1_sim; // similarities
        real nutr0_sim;
        
        for (f in 1:nfeats){
          fwise_dist_n1[f] = w[f] * (alien[f] - nutr1_pos[t,f])^2; // square the distances one by one
          fwise_dist_n0[f] = w[f] * (alien[f] - nutr0_pos[t,f])^2;
        }
        nutr1_dist = sqrt(sum(fwise_dist_n1));
        nutr0_dist = sqrt(sum(fwise_dist_n0)); // sum them and take sqrt
        
        nutr1_sim = exp(-c * nutr1_dist);
        nutr0_sim = exp(-c * nutr0_dist); // calc similarity
        
        n_rate[t] = nutr1_sim / (nutr1_sim + nutr0_sim); // rate calculated Luce's axiom
      }
      
    }
    
    // Kalman update
    
    int k;
    k = t + 1; // shifting index, so we can go back to initial values
    
    // Danger axis
    if (dangerous[t] == 1){
      // update position for d1
      dang1_K[k] = dang1_P[k-1] / (dang1_P[k-1] + R); // Kalman Gain
      
      for (f in 1:nfeats){
        dang1_pos[k,f] = dang1_pos[k-1,f] + dang1_K[k] * (alien[f] - dang1_pos[k-1,f]); // pos update
      }
      
      dang1_P[k] = (1 - dang1_K[k]) * dang1_P[k-1]; // update P
      
      // repeat pos, K, and P for d0
      dang0_K[k] = dang0_K[k-1];
      dang0_pos[k,] = dang0_pos[k-1,];
      dang0_P[k] = dang0_P[k-1];
      
    } else {
      // update position for d0
      dang0_K[k] = dang0_P[k-1] / (dang0_P[k-1] + R); // Kalman Gain
      
      for (f in 1:nfeats){
        dang0_pos[k,f] = dang0_pos[k-1,f] + dang0_K[k] * (alien[f] - dang0_pos[k-1,f]); // pos update
      }
      
      dang0_P[k] = (1 - dang0_K[k]) * dang0_P[k-1]; // update P
      
      // repeat pos, K, and P for d1
      dang1_K[k] = dang1_K[k-1];
      dang1_pos[k,] = dang1_pos[k-1,];
      dang1_P[k] = dang1_P[k-1];
      
    }
    
    // Nutrtion axis
    if (nutricious[t] == 1){
      // update position for n1
      nutr1_K[k] = nutr1_P[k-1] / (nutr1_P[k-1] + R); // Kalman Gain
      
      for (f in 1:nfeats){
        nutr1_pos[k,f] = nutr1_pos[k-1,f] + nutr1_K[k] * (alien[f] - nutr1_pos[k-1,f]); // pos update
      }
      
      nutr1_P[k] = (1 - nutr1_K[k]) * nutr1_P[k-1]; // update P
      
      // repeat pos, K, and P for n0
      nutr0_K[k] = nutr0_K[k-1];
      nutr0_pos[k,] = nutr0_pos[k-1,];
      nutr0_P[k] = nutr0_P[k-1];
      
    } else {
      // update position for n0
      nutr0_K[k] = nutr0_P[k-1] / (nutr0_P[k-1] + R); // Kalman Gain
      
      for (f in 1:nfeats){
        nutr0_pos[k,f] = nutr0_pos[k-1,f] + nutr0_K[k] * (alien[f] - nutr0_pos[k-1,f]); // pos update
      }
      
      nutr0_P[k] = (1 - nutr0_K[k]) * nutr0_P[k-1]; // update P
      
      // repeat pos, K, and P for n1
      nutr1_K[k] = nutr1_K[k-1];
      nutr1_pos[k,] = nutr1_pos[k-1,];
      nutr1_P[k] = nutr1_P[k-1];
      
    }
    
  }
}

// The model to be estimated.

model {
  // Priors
  target += normal_lpdf(logit_c | 0, 1.5); // normal prior for logit c
  target += normal_lpdf(log_R | 0, 1); // normal prior for log R
  target += dirichlet_lpdf(w | w_prior_values); // kappa param implicitly = 1
  
  // likelihood function
  
  target += bernoulli_lpmf(resp_d | d_rate); // prob of response of danger given rate for danger
  target += bernoulli_lpmf(resp_n | n_rate); // prob of response of nutrition given rate for nutrition
}

////////
//   The rates for dang and nutr must be calculated for each trial in transformed parameters



generated quantities{
  // priors
  real log_R_prior;
  real R_prior;
  real logit_c_prior;
  real c_prior;
  simplex[nfeats] w_prior;
  
  log_R_prior = normal_rng(0,1);
  R_prior = exp(log_R_prior);
  
  logit_c_prior = normal_rng(0, 1.5);
  c_prior = inv_logit(logit_c_prior) * 4;
  
  w_prior = dirichlet_rng(w_prior_values);
  
  // Is it okay, if we skip prior predictions and posterior predictions? :)
  
}


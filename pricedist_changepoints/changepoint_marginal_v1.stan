functions {
  matrix create_histogram_cumulative(matrix histogram)
  {
      int n_time_points = rows(histogram);
      matrix[n_time_points, cols(histogram)] result;
      
      result[1] = histogram[1];
      if( sum(histogram) == 0 ) {
          reject("Row 1 in quantity histogram is all zeroes. Please exclude such rows.");
      }
      
      for (t in 2:n_time_points) {
          result[t] = result[t - 1] + histogram[t];
            if ( sum(histogram[t]) == 0 ) {
              reject("Row ", t, " in quantity histogram is all zeroes. Please exclude such rows.");
            }

      }
      return result;
  }

  vector extract_segment_histogram(int t_start, int t_end, matrix histogram_cumulative)
  {
      int n_time_points = rows(histogram_cumulative);
      int n_price_points = cols(histogram_cumulative);
      row_vector[n_price_points] start_vals;
      row_vector[n_price_points] end_vals;
      row_vector[n_price_points] result;
      int t_prev = t_start - 1;

      if (t_prev < 1) {
        start_vals = rep_row_vector(0.0, n_price_points);
      } else {
        start_vals = histogram_cumulative[t_prev];
      }

      end_vals = (t_end <= n_time_points) ? histogram_cumulative[t_end] :  histogram_cumulative[n_time_points];
      result = end_vals - start_vals;
  
      return to_vector(result);
  }

  real compute_segment_loglik(int t_start, int t_end, matrix histogram_cumulative)
  {
      int n_time_points = cols(histogram_cumulative);
      vector[n_time_points] histogram = extract_segment_histogram(t_start, t_end, histogram_cumulative);
      real total_qty = sum(histogram);
      vector[n_time_points] log_probs;
  
      for (i in 1:cols(histogram_cumulative)) {
          log_probs[i] = histogram[i] > 0 ? log(histogram[i] / total_qty) : 0.0;
      }

      return total_qty > 0 ? dot_product(histogram, log_probs) : 0.0;
  }

  matrix compute_segments_loglik(matrix histogram_cumulative)
  {
      int n_time_points = rows(histogram_cumulative);
      matrix[n_time_points, n_time_points] segments_loglik;
  
      for (t1 in 1:n_time_points) {
          for (t2 in t1:n_time_points) {
              segments_loglik[t1, t2] = compute_segment_loglik(t1, t2, histogram_cumulative);
          }
          for (t2 in 1:(t1 - 1)) {
              segments_loglik[t1, t2] = negative_infinity();
          }
      }
  
      return segments_loglik;
  }

  real compute_change_magnitude(vector histogram_prev, vector histogram_curr, vector price_points)
  {
      real mean_prev = dot_product(histogram_prev, price_points) / sum(histogram_prev);
      real mean_curr = dot_product(histogram_curr, price_points) / sum(histogram_curr);
      return abs(mean_prev - mean_curr);
  }

  vector compute_change_magnitudes(matrix histogram_cumulative, vector price_points)
  {
      int n_time_points = rows(histogram_cumulative);
      int n_price_points = cols(histogram_cumulative);
      vector[n_time_points] deltas;
      
      deltas[1] = 0.0;
      for (t in 2:n_time_points) {
          vector[n_price_points] histogram_prev = extract_segment_histogram(t - 1, t - 1, histogram_cumulative);
          vector[n_price_points] histogram_curr = extract_segment_histogram(t,     t,     histogram_cumulative);
          deltas[t] = compute_change_magnitude(histogram_prev, histogram_curr, price_points);
      }
  
      return deltas;
  }

  vector compute_lp_change_prior(vector change_magnitude, real prior_mu, real prior_sigma)
  {
      int T = num_elements(change_magnitude);
      vector[T] lp;

      for (t in 1:T) {
          lp[t] = normal_lpdf( change_magnitude[t] | prior_mu, prior_sigma);
      }
  
      return lp;
  }

  real compute_marginal_loglik(vector cp_probs_raw, matrix segments_loglik, vector lp_change_prior)
  {
        int n_time_points = dims(segments_loglik)[1];
        int n_cp = n_time_points - 1;
        real epsilon = 1e-12;
        vector[n_cp] cp_probs = fmin(fmax(cp_probs_raw, epsilon), 1 - epsilon);
        vector[n_cp] lp_cp = log(cp_probs);
        vector[n_cp] lp1m_cp = log1m(cp_probs);
        vector[n_time_points + 1] marginal_loglik;
    
        marginal_loglik[1] = 0.0;
    
        for (t2 in 2:(n_time_points + 1)) {
            vector[t2 - 1] path_lls;
    
            for (t1 in 1:(t2 - 1)) {
                real prev_marginal = marginal_loglik[t1];
    
                real lp_cp_cur = t1 > 1 ? lp_cp[t1 - 1] : 0.0;
                real lp_no_cp_cur = t1 < (t2 - 2) ? sum(lp1m_cp[t1:(t2 - 2)]) : 0.0;
    
                real ll_segment = segments_loglik[t1, t2 - 1];
                real lp_segment_change = t1 > 1 ? lp_change_prior[t2 - 1] : 0.0;
    
                path_lls[t1] = prev_marginal + lp_cp_cur + lp_no_cp_cur + ll_segment + lp_segment_change;
            }
    
            marginal_loglik[t2] = log_sum_exp(path_lls);
        }
    
        return marginal_loglik[n_time_points + 1];
  }

}


data {
  int<lower=1> n_time_points;
  int<lower=1> n_price_points;
  matrix[n_time_points, n_price_points] histogram;
  vector[n_price_points] price_points;

}

transformed data {
  matrix[n_time_points, n_price_points] histogram_cumulative = create_histogram_cumulative(histogram);
  matrix[n_time_points, n_time_points] segments_loglik = compute_segments_loglik(histogram_cumulative);
  vector[n_time_points] change_magnitudes = compute_change_magnitudes(histogram_cumulative, price_points);
}

parameters {
  vector<lower=0, upper=1>[n_time_points-1] cp_probs;

  // Prior hyperparameters
  real<lower=1> prior_change_mu;
  real<lower=0.0001> prior_change_sigma;
}

transformed parameters {
  vector[n_time_points] lp_change_prior = compute_lp_change_prior(change_magnitudes, prior_change_mu, prior_change_sigma);
}

model {
  // Optionally add prior over cp_probs_raw if desired, e.g.:
  cp_probs ~ beta(1, 5);
  prior_change_mu ~ normal(0, .5);
  prior_change_sigma ~ exponential(1);

  target += compute_marginal_loglik(cp_probs, segments_loglik, lp_change_prior);
}

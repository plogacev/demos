functions {
  
  real normal_overlap(real mu1, real sigma1, real mu2, real sigma2) {
      real d = abs(mu1 - mu2) / sqrt(square(sigma1) + square(sigma2));
      return 2 * normal_cdf(d / 2 | 0, 1) - 1;
  }
  
  array[] vector local_weighted_histograms(int t_mid, int window_size, array[,] int histogram, vector log_ncp_probs)
  {
      int n_time_points = dims(histogram)[1];
      int n_price_points = dims(histogram)[2];
      
      int window_size_left  = min(window_size, t_mid-1);
      int window_size_right = min(window_size, n_time_points-t_mid);

      real epsilon = 1e-6;
      vector[n_price_points] left_segment_histogram = rep_vector( epsilon, n_price_points );
      vector[n_price_points] right_segment_histogram = rep_vector( epsilon, n_price_points );

      if (window_size_left > 0) {
          real log_weight = 0.0;
          for ( s in 0:(window_size_left-1) ) {
              int t_cur = t_mid - s;
              log_weight += log_ncp_probs[t_cur];
              left_segment_histogram += to_vector(histogram[t_cur]) * exp(log_weight);
          }
      }

      if (window_size_right > 0) {
          real log_weight = 0.0;
          for (s in 1:window_size_right) {
              int t_cur = t_mid + s;
              right_segment_histogram += to_vector(histogram[t_cur]) * exp(log_weight);
              if ( s < window_size_right ) {
                  log_weight += log_ncp_probs[t_cur];
              }
          }
      }

      array[2] vector[n_price_points] results;
      results[1] = left_segment_histogram;
      results[2] = right_segment_histogram;

      return results;
  }

  // Computes cumulative sum of the histogram over time
  array[,] int create_histogram_cumulative(array[,] int histogram)
  {
      int n_time_points = dims(histogram)[1];
      int n_price_points = dims(histogram)[2];
      array[n_time_points, n_price_points] int result;
      
      // Initialize with first row
      result[1] = histogram[1];

      // Reject if all values are zero in the entire histogram
      if( sum(histogram[1]) == 0 ) {
          reject("Row 1 in quantity histogram is all zeroes. Please exclude such rows.");
      }
      
      // Compute cumulative sum row-wise
      for (t in 2:n_time_points) {
          for (p in 1:n_price_points)
            result[t][p] = result[t - 1][p] + histogram[t][p];
          
          // Check for empty histogram rows (no prices at all)
          if ( sum(histogram[t]) == 0 ) {
            reject("Row ", t, " in quantity histogram is all zeroes. Please exclude such rows.");
          }

      }
      return result;
  }

  // Extracts the histogram for a segment [t_start, t_end]
  vector extract_segment_histogram(int t_start, int t_end, array[,] int histogram_cumulative)
  {
      int n_time_points = dims(histogram_cumulative)[1];
      int n_price_points = dims(histogram_cumulative)[2];
      array[n_price_points] int start_vals;
      array[n_price_points] int end_vals;
      array[n_price_points] int result;
      
      
      // Retrieve cumulative before the start
      int t_prev = t_start - 1;
      if (t_prev < 1) {
          for (p in 1:n_price_points)
              start_vals[p] = 0;
      } else {
          start_vals = histogram_cumulative[t_prev];
      }

      // Compute end cumulative value
      end_vals = (t_end <= n_time_points) ? histogram_cumulative[t_end] :  histogram_cumulative[n_time_points];
      
      // Segment = difference between cumulative ends
      for (p in 1:n_price_points)
          result[p] = end_vals[p] - start_vals[p];

      return to_vector(result);
  }

  // Computes segment log-likelihood under multinomial model
  real compute_segment_loglik(int t_start, int t_end, array[,] int histogram_cumulative)
  {
      int n_price_points = dims(histogram_cumulative)[2];
      vector[n_price_points] histogram = extract_segment_histogram(t_start, t_end, histogram_cumulative);
      real total_qty = sum(histogram);
      vector[n_price_points] log_probs;
  
      // Compute log-probabilities for multinomial
      for (i in 1:n_price_points) {
          log_probs[i] = histogram[i] > 0 ? log(histogram[i] / total_qty) : 0.0;
      }

      // Return segment log-likelihood
      return total_qty > 0 ? dot_product(histogram, log_probs) : 0.0;
  }

  // Precomputes all segment log-likelihoods for dynamic programming algo
  matrix compute_segments_loglik(array[,] int histogram_cumulative)
  {
      int n_time_points = dims(histogram_cumulative)[1];
      matrix[n_time_points, n_time_points] segments_loglik;
  
      // Compute upper triangle log-likelihoods
      for (t1 in 1:n_time_points) {
          for (t2 in t1:n_time_points) {
              segments_loglik[t1, t2] = compute_segment_loglik(t1, t2, histogram_cumulative);
          }
      }

      // Set lower triangle to -Inf (not valid segments)
      for (t1 in 1:n_time_points) {
          for (t2 in 1:(t1 - 1)) {
              segments_loglik[t1, t2] = negative_infinity();
          }
      }
  
      return segments_loglik;
  }

  // Computes segment-by-segment difference statistic (e.g., mean price here) as a proxy for change magnitude
  // to-do: implement it by segment (with reasonable pruning)
  real compute_change_magnitude(vector histogram_prev, vector histogram_curr, vector price_points)
  {
      // Retrieve previous and current segment
      real mean_prev = dot_product(histogram_prev, price_points) / sum(histogram_prev);
      real mean_curr = dot_product(histogram_curr, price_points) / sum(histogram_curr);
      
      // Change magnitude is absolute mean difference
      return abs(mean_prev - mean_curr);
  }

  // Computes change magnitude between adjacent time steps
  // to-do: think of a more robust metric than the mean, something distributional
  // to-do: apply shrinking through a prior, taking into account the sample sizes
  vector compute_change_magnitudes(array[,] int histogram, array[,] int histogram_cumulative, vector price_points, int window_size, vector lp_ncp)
  {
      int n_time_points = dims(histogram_cumulative)[1];
      int n_price_points = dims(histogram_cumulative)[2];
      vector[n_time_points-1] deltas;
      
      // Compare each time step with previous
      // to-do: Compute a weighted local window like in the window algorithm
      for (t in 1:(n_time_points-1)) {
          array[2] vector[n_price_points] local_hist = local_weighted_histograms(t, window_size, histogram, lp_ncp);
          vector[n_price_points] left_segment_histogram = local_hist[1];
          vector[n_price_points] right_segment_histogram = local_hist[2];

          deltas[t] = compute_change_magnitude(left_segment_histogram, right_segment_histogram, price_points);
      }
      
      return deltas;
  }

  real changepoint_magnitude_prior(real x, real percentile_5, real percentile_50) {
    real k = log(19.0) / (percentile_50 - percentile_5); 
    return -log1p_exp(-k * (x - percentile_50));
  }
  
  // Applies change magnitude prior to the change magnitude
  // These priors serve as a penalty or encouragement for placing changepoints
  vector compute_lp_change_prior(vector change_magnitude, vector lp_cp, real change_magnitude_min, real change_magnitude_typical)
  {
      int T = num_elements(change_magnitude);
      vector[T] lp;

      for (t in 1:T) {
              lp[t] = changepoint_magnitude_prior(change_magnitude[t], change_magnitude_min, change_magnitude_typical);
      }

      return lp;
  }

  // Main forward pass algorithm: computes marginal log-likelihood via dynamic programming
  // Each time step t2 accumulates total log-probabilities over all segmentations ending at t2
  real compute_marginal_loglik(vector lp_cp, vector lp_ncp, matrix segments_loglik, vector lp_change_prior)
  {
        int n_time_points = dims(segments_loglik)[1];
        int n_cp = n_time_points - 1;

        vector[n_time_points + 1] marginal_loglik;

        // base case: empty segment
        marginal_loglik[1] = 0.0;
    
        // Dynamic programming over segment ends
        for (t2 in 2:(n_time_points + 1)) {
            vector[t2 - 1] path_lls;
    
            // Iterate over all previous segmentations ending at t1
            for (t1 in 1:(t2 - 1)) {
                real prev_marginal = marginal_loglik[t1];
    
                real lp_cp_cur = t1 > 1 ? lp_cp[t1 - 1] : 0.0;
                real lp_no_cp_cur = t1 < (t2 - 2) ? sum(lp_ncp[t1:(t2 - 2)]) : 0.0;
                real lp_path = lp_cp_cur + lp_no_cp_cur;

                real lp_segment_change = t1 > 1 ? lp_change_prior[t1 - 1] : 0.0;

                real ll_segment = segments_loglik[t1, t2 - 1];
    
                // Total log-prob for this segmentation path
                path_lls[t1] = prev_marginal + lp_path + ll_segment + lp_segment_change;
            }
    
            // Aggregate over paths to compute marginal
            marginal_loglik[t2] = log_sum_exp(path_lls);
        }
    
        // Final log-marginal likelihood
        return marginal_loglik[n_time_points + 1];
  }

}


data {
  int<lower=1> n_time_points;
  int<lower=1> n_price_points;
  array[n_time_points, n_price_points] int histogram;
  vector[n_price_points] price_points;
  
  int change_window_size;
  real prior_cp_probs_one;
  real prior_change_magnitude_min;
  real prior_change_magnitude_typical;
}

transformed data {
  array[n_time_points, n_price_points] int histogram_cumulative = create_histogram_cumulative(histogram);
  matrix[n_time_points, n_time_points] segments_loglik = compute_segments_loglik(histogram_cumulative);
}

parameters {
  vector<upper=0>[n_time_points-1] lp_cp;
  real<upper=0> lperc_cp_one;


}

transformed parameters {
  real<lower=0> prior_change_skew = 0;

  vector<upper=0>[n_time_points-1] lp_ncp = log1m_exp(lp_cp);
  vector[n_time_points-1] change_magnitudes = compute_change_magnitudes(histogram, histogram_cumulative, price_points, change_window_size, lp_ncp);
  vector[n_time_points-1] lp_change_prior = compute_lp_change_prior(change_magnitudes, lp_cp, prior_change_magnitude_min, prior_change_magnitude_typical);

}

model {

  lperc_cp_one ~ normal(-5, 10);
  for (i in 1:(n_time_points-1)) {
    target += log_sum_exp( lperc_cp_one + lp_cp[i], log1m_exp(lperc_cp_one) + lp_ncp[i] );
  }

  target += compute_marginal_loglik(lp_cp, lp_ncp, segments_loglik, lp_change_prior);
}

# Self-Contained Data Science Projects 

## 1. Time Series Elasticity Analysis

**Demonstration of a NumPyro-based Bayesian time series model for sales data analysis.**
A synthetic sales dataset is constructed with latent growth, seasonal patterns, and a random walk component to simulate external influences.
Sales are modeled as a Poisson process with price elasticity effects, ensuring realistic demand shifts. A flexible Bayesian model, 
implemented in numpyro, is used to estimate the underlying components of the sales time series. Inference is performed using MCMC sampling. 
Despite structural differences, the model is flexible enough to meaningfully separate conceptually distinct components of 
the sales time series, such as trend, monthly and weekly seasonality, as well as price elasticity of demand. 
The results confirm that the model successfully reconstructs these components, demonstrating its suitability for the analysis of sales data.

🔗 [Full Project Details](time_series_analysis_1/README.md)  


## 2. Distribution-based Change Point Detection

**Bayesian changepoint model for detecting structural changes in pricing regimes.**
Different implementations of a Bayesian changepoint detection model for detecting structural changes in the data-generating process, such as 
different pricing regimes. Estimation is based on price histograms over time when the pricing of a product undergoes a meaningful change — such as list price changes, promotions, or changes in the availability
of discounts.

🔗 [Full Project Details](pricedist_changepoints/README.md)  

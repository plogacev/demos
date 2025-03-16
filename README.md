# Small Self-Contained Data Science Projects 

## Section 1: Time Series Elasticity Analysis

A synthetic sales dataset is constructed with latent growth, seasonal patterns, and a random walk component to simulate external influences. Sales are modeled as a Poisson process with price elasticity effects, ensuring realistic demand shifts. A flexible Bayesian model, implemented in numpyro, is used to estimate the underlying components of the sales time series. Inference is performed using MCMC sampling. Despite structural differences, the model is flexible enough to meaningfully separate conceptually distinct components of the sales time series, such as trend, monthly and weekly seasonality, as well as price elasticity of demand. The results confirm that the model successfully reconstructs these components, demonstrating its suitability for sales forecasting.
[Project A Details](time_series_analysis_1/README.md)

## Section 2: Project B
[Project B Details](project_b.md)
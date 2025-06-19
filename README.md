*This document is best viewed [here](https://github.com/plogacev/case_studies) in HTML rather than in markdown format.*


# Self-Contained Data Science Projects 

## 1. Time Series Elasticity Analysis

**NumPyro-based Bayesian time series model for sales data analysis.**
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

🔗 [Full Project Details](pricedist_changepoints/README.html)

## 3. Price Distribution Elicitation from an LLM

**Two-stage price distribution elicitation pipeline.**
Demonstrates a two-stage pipeline that uses large language models (LLMs) to infer reasonable price distributions for consumer products when competitor pricing data is insufficient or historical sales data is sparse.
It extracts structured product attributes from free-text titles and descriptions (via GPT-3.5-turbo), then uses GPT-4o to elicit a retail price range (minimum, typical, maximum) based
on those attributes and regional context. The elicited price ranges are used to reverse-engineer log-normal distribution parameters, which are validated against actual prices scraped from Mercado Libre.
The result is a calibrated, probabilistic estimate of product pricing grounded in LLM-elicited knowledge.

🔗 [Full Project Details](price_distribution_elicitation/price_distribution_elicitation.html)

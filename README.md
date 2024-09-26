# The background
Mathematically, the calibration problem of the volatility model is an inverse problem of determining the model parameters. Through calibration, the derivative product price calculated by the model is consistent with the market price of the product. Since the constant volatility assumption in the Black-Scholes model [5] cannot reflect the volatility smile and term structure of volatility observed in the market, in the past few decades, researchers have been trying to calibrate the market with richer volatility models, including stochastic local volatility models and models with jumps.

# The components

- yutian_initial

calculates the initial joint distribution of two variables, 𝑆 and 𝑧, using specified parameters

- finite_difference_mat_b2

The function aims to: Create finite difference matrices for the asset prices (S) and a secondary variable (z), which could represent factors like volatility. 

- adi_p_b

calculating option price changes over time using the Alternating Directional Implicit (ADI) method. The function calls finite_difference_mat_b2 to create finite difference matrices that represent the spatial derivatives for the pricing model. This is crucial for the ADI scheme, which involves solving a system of equations based on these derivatives. The function iterates over time steps (nt) to update the price grid p using the ADI method.
Within each time step, it computes terms like F0p, F1, F2p, which represent the contributions to the pricing evolution based on the model parameters and current prices.
The matrix A is formed and then used to derive the intermediate value B for the ADI calculations.

- inpaint_nans

 is a implementation for filling in NaN values in an array using various numerical methods

- OptionPrice2_put

is part of a numerical method to price European put options using finite difference techniques. 
The function aims to calculate the price of European put options based on the Black-Scholes model or a similar framework, utilizing a grid-based approach for numerical approximation.

- initial_by_bisection

is designed to compute an initial price distribution for a financial model. It uses a bisection method to iteratively adjust a time step until the total probability integrates to 1. Here’s a breakdown of its components and functionality:

- Calibration_FinitePoint4

Calculate the leverage surface and implied volatility surface for options across various maturities.
Use market data to calibrate a model that estimates how options prices vary with changes in underlying asset prices and maturities.

- test_estimate_performance

Calibrate a model that estimates implied volatility for European options based on market data.
Visualize the results by comparing the model's implied volatility surface with market-observed values.

# Travelers policyholder call

**Goal:** Build a model to predict how often policyholders call, to help CloverShield Insurance plan call-center resources.  
Below is the **plan**; implementation is still in progress.

## Data processing (planned)

1. Split into **NonAuto** and **Auto**, and prepare each segment separately (drop fields that do not apply or are redundant for that segment).

2. Split train into **train** and **validation**.

3. **KNN imputation** for categoricals with missing values.  
   - Train the imputer on **NonAuto train** only; apply to NonAuto train, val, and test.  
   - Train the imputer on **Auto train** only; apply to Auto train, val, and test.

## Models (planned)

**Baselines**

- **Zero-inflated Poisson (ZIP)** — count baseline that adds a zero-inflation layer for excess zeros; the Poisson count part still assumes variance equals mean (often too strict if counts remain overdispersed after accounting for zeros).

- **Negative binomial regression** — handles overdispersion (variance > mean); better for heavy tails and bursty counts.

- **Zero-inflated negative binomial (ZINB)** — if there are many zeros from a separate process than the main count process.

**More advanced**

- **Hurdle (two stages)**  
  - Stage 1: predict P(Y > 0).  
  - Stage 2: on rows with Y > 0, predict E[Y | Y > 0], e.g. negative binomial or **LightGBM with Tweedie** objective.  
  - Combined: ŷ = P(Y > 0) × E[Y | Y > 0].

- **Single-stage boosting** — LightGBM / XGBoost with **Tweedie** objective (no hurdle), for comparison.

## Experiment order (planned)
1. Zero-inflated Poisson (ZIP)
    https://www.statsmodels.org/dev/generated/statsmodels.discrete.count_model.ZeroInflatedPoisson.html
2. Zero-inflated negative binomial (ZINB)  
    Statistical ZINB regression: https://www.statsmodels.org/dev/generated/statsmodels.discrete.count_model.ZeroInflatedNegativeBinomialP.html
    Bayesian ZINB modeling: https://www.pymc.io/projects/docs/en/stable/api/distributions/generated/pymc.ZeroInflatedNegativeBinomial.html

3. Hurdle: binary + negative binomial  
    statsmodels negative binomial: https://www.statsmodels.org/stable/generated/statsmodels.discrete.discrete_model.NegativeBinomial.html

4. Hurdle: binary + boosting (with NB loss)
    xgboost negative-binomial: https://xgboost-distribution.readthedocs.io/en/latest/api/xgboost_distribution.XGBDistribution.html

## Metrics (planned)

- **RMSE** on validation for comparing predictions.  
- **AIC** for classical models where it applies.

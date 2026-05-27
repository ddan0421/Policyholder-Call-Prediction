# Policyholder Call Prediction

Project Status: Work in Progress

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

- **Poisson GLM** — simplest count regression with a log link; assumes variance equals mean and ignores excess zeros. Used as a reference benchmark for ZIP/ZINB.

- **Zero-inflated Poisson (ZIP)** — count baseline that adds a zero-inflation layer for excess zeros; the Poisson count part still assumes variance equals mean (often too strict if counts remain overdispersed after accounting for zeros).

- **Zero-inflated negative binomial (ZINB)** — if there are many zeros from a separate process than the main count process.

**More advanced**

- **Hurdle (two stages)**  
  - Stage 1: predict P(Y > 0).  
  - Stage 2: on rows with Y > 0, predict E[Y | Y > 0], e.g. negative binomial or **boosting with NB loss** objective.  
  - Combined: ŷ = P(Y > 0) × E[Y | Y > 0].

### Zero-inflated vs hurdle: how each approach treats zeros

Both approaches address the same problem (lots of zeros + overdispersed positive counts), but they make **different assumptions about where the zeros come from**.

#### Zero-inflated (ZIP / ZINB)

A zero-inflated model is **two regressions fit jointly** on the same features X:

1. **Inflation logit** — probability of being a structural zero ("never going to call"):

$$
\pi(X) = \mathrm{logit}^{-1}\!\Big(\beta^{\text{infl}}_0 + \sum_j \beta^{\text{infl}}_j X_j\Big)
$$

2. **Count regression** — conditional mean of the count process (Poisson for ZIP, NB for ZINB):

$$
\lambda(X) = \exp\!\Big(\beta^{\text{cnt}}_0 + \sum_j \beta^{\text{cnt}}_j X_j\Big)
$$

A zero in the data can come from **two sources**:

$$
P(Y=0 \mid X) \;=\; \underbrace{\pi(X)}_{\text{structural zero}} \;+\; \underbrace{(1-\pi(X))\cdot P_{\text{count}}(Y=0 \mid X)}_{\text{count zero (sampled 0)}}
$$

For $k > 0$:

$$
P(Y=k \mid X) \;=\; (1-\pi(X))\cdot P_{\text{count}}(Y=k \mid X)
$$

The count distribution is allowed to **also produce zeros**. ZIP/ZINB makes the most sense when the data contains genuinely two sub-populations: a "dormant" group that essentially never calls, and an "active" Poisson/NB group whose count occasionally lands on 0.

#### Hurdle

A hurdle model splits zeros and positives **cleanly**:

1. **Stage 1 (binary)** — logistic regression on Y > 0. All zeros come from this stage only.
2. **Stage 2 (zero-truncated count)** — fit on rows with Y > 0 only, using a count distribution that **cannot produce zeros**.

$$
P(Y=0 \mid X) \;=\; \pi(X)
$$

$$
P(Y=k \mid X) \;=\; (1-\pi(X))\cdot \frac{P_{\text{count}}(k \mid X)}{1 - P_{\text{count}}(0 \mid X)}, \qquad k > 0
$$

The two stages are estimated **independently**, and stage 2 can be a different model family (e.g. logistic for stage 1, NB or boosted-NB for stage 2).

#### Side-by-side

| | ZIP / ZINB | Hurdle |
|---|---|---|
| Sources of zero | Two: structural + count-process zero | One: only the binary stage |
| Count distribution | Can produce zeros | **Zero-truncated** (no zeros allowed) |
| Estimation | Two parts fit jointly via MLE | Two parts fit independently |
| Best fit when | Data is a mixture of "never callers" + an active count population | "Did the customer call at all?" is a different decision from "how many times?" |

## Experiment order (planned)
1. Poisson GLM (reference baseline)
    https://www.statsmodels.org/dev/generated/statsmodels.genmod.generalized_linear_model.GLM.html

2. Zero-inflated Poisson (ZIP)
    https://www.statsmodels.org/dev/generated/statsmodels.discrete.count_model.ZeroInflatedPoisson.html

3. Zero-inflated negative binomial (ZINB)  
    Statistical ZINB regression: https://www.statsmodels.org/dev/generated/statsmodels.discrete.count_model.ZeroInflatedNegativeBinomialP.html
    Bayesian ZINB modeling: https://www.pymc.io/projects/docs/en/stable/api/distributions/generated/pymc.ZeroInflatedNegativeBinomial.html

4. Hurdle: binary + negative binomial  
    statsmodels negative binomial: https://www.statsmodels.org/stable/generated/statsmodels.discrete.discrete_model.NegativeBinomial.html

5. Hurdle: binary + boosting (with NB loss)
    xgboost negative-binomial: https://xgboost-distribution.readthedocs.io/en/latest/api/xgboost_distribution.XGBDistribution.html

## Metrics (planned)

- **RMSE** on validation for comparing predictions.  
- **AIC** for classical models where it applies.  
- **Normalized Gini** — rank-based metric: sort policies by predicted call counts and measure how well actual calls concentrate at the top of the ranking. Range is roughly $[-1, 1]$; **1.0 = perfect ranking, 0 = random, negative = anti-correlated**.

### Normalized Gini

Sort rows by predicted call counts (descending; highest-risk first). Let $N$ be the number of observations and $a_i$ the actual call count at rank $i$ (so $a_1$ is at the highest predicted, $a_N$ at the lowest predicted). Define

$$
G(\hat y) = \frac{1}{N}\left(\frac{\sum_{k=1}^{N} \sum_{i=1}^{k} a_i}{\sum_{i=1}^{N} a_i} - \frac{N+1}{2}\right)
$$

Then the normalized Gini is

$$
\text{Normalized Gini} = \frac{G(\hat y)}{G(y)}
$$

where $G(y)$ is the same quantity computed with predictions equal to the true counts (perfect ranking).

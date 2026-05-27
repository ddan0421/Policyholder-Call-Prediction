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

### Count distributions: Poisson vs Negative Binomial

All count models below plug one of these two distributions into the count part of the model. The difference between them is entirely about how much **variance** they allow for a given mean.

#### Poisson

A single-parameter count distribution with rate μ = E[Y]:

$$
P(Y = k \mid \mu) = \frac{e^{-\mu} \mu^{k}}{k!}, \quad k = 0, 1, 2, \ldots
$$

Key property: **variance equals mean** (Var(Y) = μ). This is restrictive — real-world count data is often more spread out than Poisson allows.

In a regression, μ is linked to features through a log link:

$$
\mu(X) = \exp\left(\beta_0 + \sum_j \beta_j X_j\right)
$$

#### Negative Binomial

The Negative Binomial distribution has two common parameterizations. They describe the **same family of distributions**, just from different angles.

##### (a) Classical probability-theory form — "trials until k successes"

Counts the number of trials X needed to see exactly k successes, where each trial succeeds independently with probability p:

$$
P(X = x \mid k, p) = \frac{(x - 1)!}{(k - 1)! (x - k)!} p^{k} (1 - p)^{x - k}, \quad x = k, k+1, k+2, \ldots
$$

- x = number of trials at which the k-th success occurs (x ≥ k)
- p = probability of success on each trial
- k = (integer) number of successes you are waiting for
- Mean: μ<sub>X</sub> = k / p
- Standard deviation: σ<sub>X</sub> = √(k(1 − p)) / p

This is the form you usually see in a probability textbook (sometimes called the **Pascal** distribution). It generalizes the geometric distribution (which is the k = 1 case).

##### (b) NB2 regression form — "count with mean μ and dispersion α"

The form used by statsmodels (and what fits naturally into GLM-style regression). Let Y be the **count itself** (e.g. the number of failures before the k-th success, or just an integer outcome ≥ 0):

$$
P(Y = y \mid \mu, \alpha) = \frac{\Gamma(y + 1/\alpha)}{\Gamma(1/\alpha) \, y!} \left(\frac{1/\alpha}{1/\alpha + \mu}\right)^{1/\alpha} \left(\frac{\mu}{1/\alpha + \mu}\right)^{y}, \quad y = 0, 1, 2, \ldots
$$

- Mean: E[Y] = μ
- Variance: Var(Y) = μ + α μ² — strictly larger than the mean whenever α > 0.
- As α → 0, NB collapses back to Poisson.

##### How the two forms relate

They describe the same distribution, just shifted and reparameterized:

- The classical form counts **trials** X ≥ k. The NB2 form counts **failures** Y = X − k ≥ 0.
- The (k, p) and (μ, α) parameters are linked by:

$$
\alpha = \frac{1}{k}, \quad \mu = \frac{k(1 - p)}{p}, \quad p = \frac{1}{1 + \alpha \mu}
$$

So the two forms differ in three practical ways:

1. **Support.** Classical: X ∈ {k, k+1, …}. NB2: Y ∈ {0, 1, 2, …} — required for a count regression where outcomes can be 0.
2. **Parameters.** Classical uses (k, p) — both interpretable as a "stopping rule" on Bernoulli trials, with k an integer. NB2 uses (μ, α) — directly the **mean** and a **dispersion** knob, with 1/α allowed to be any positive real (continuous).
3. **Use case.** Classical is convenient when the problem really is "trials until k successes". NB2 is convenient for regression: μ is connected to features via a log link (μ(X) = exp(β₀ + Σ β<sub>j</sub> X<sub>j</sub>)), and α captures overdispersion in a single coefficient that can be tested and interpreted.

In this project, only the NB2 form is used — it lets the model learn how μ varies with policyholder features while a single dispersion α absorbs the extra variance Poisson cannot handle.

#### Side-by-side

| | Poisson | Negative Binomial |
|---|---|---|
| Parameters | μ | μ, α |
| Mean | μ | μ |
| Variance | μ | μ + α μ² |
| Handles overdispersion | No | Yes |
| Used as P_count in | Poisson GLM, ZIP, hurdle stage 2 (Poisson) | NB regression, ZINB, hurdle stage 2 (NB / boosted-NB) |

### Zero-inflated vs hurdle: how each approach treats zeros

Both approaches address the same problem (lots of zeros + overdispersed positive counts), but they make **different assumptions about where the zeros come from**.

#### Zero-inflated (ZIP / ZINB)

A zero-inflated model is **two regressions fit jointly** on the same features X:

1. **Inflation logit** — probability of being a structural zero ("never going to call"):

$$
\pi(X) = \mathrm{logit}^{-1}\left(\beta^{\mathrm{infl}}_0 + \sum_j \beta^{\mathrm{infl}}_j X_j\right)
$$

2. **Count regression** — conditional mean of the count process (Poisson for ZIP, NB for ZINB):

$$
\lambda(X) = \exp\left(\beta^{\mathrm{cnt}}_0 + \sum_j \beta^{\mathrm{cnt}}_j X_j\right)
$$

A zero in the data can come from **two sources**:

$$
P(Y=0 \mid X) = \pi(X) + (1-\pi(X)) \cdot P_{\mathrm{count}}(Y=0 \mid X)
$$

where the first term is the **structural zero** and the second is a **count zero** (the count distribution sampled 0).

For k > 0:

$$
P(Y=k \mid X) = (1-\pi(X)) \cdot P_{\mathrm{count}}(Y=k \mid X)
$$

The count distribution is allowed to **also produce zeros**. ZIP/ZINB makes the most sense when the data contains genuinely two sub-populations: a "dormant" group that essentially never calls, and an "active" Poisson/NB group whose count occasionally lands on 0.

#### Hurdle

A hurdle model splits zeros and positives **cleanly**:

1. **Stage 1 (binary)** — logistic regression on Y > 0. All zeros come from this stage only.
2. **Stage 2 (zero-truncated count)** — fit on rows with Y > 0 only, using a count distribution that **cannot produce zeros**.

$$
P(Y=0 \mid X) = \pi(X)
$$

$$
P(Y=k \mid X) = (1-\pi(X)) \cdot \frac{P_{\mathrm{count}}(k \mid X)}{1 - P_{\mathrm{count}}(0 \mid X)}, \quad k > 0
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
    https://www.statsmodels.org/dev/generated/statsmodels.discrete.count_model.ZeroInflatedNegativeBinomialP.html

4. Hurdle: binary + negative binomial  
    statsmodels negative binomial: https://www.statsmodels.org/stable/generated/statsmodels.discrete.discrete_model.NegativeBinomial.html

5. Hurdle: binary + boosting (with NB loss)
    xgboost negative-binomial: https://xgboost-distribution.readthedocs.io/en/latest/api/xgboost_distribution.XGBDistribution.html

### Bayesian alternative with PyMC (not demonstrated here)

All of the count models above can also be fit in a Bayesian framework with PyMC. This project sticks with the frequentist statsmodels implementations, but PyMC offers ready-made distributions for each of the count models discussed:

- Zero-inflated Poisson: https://www.pymc.io/projects/docs/en/stable/api/distributions/generated/pymc.ZeroInflatedPoisson.html
- Zero-inflated negative binomial: https://www.pymc.io/projects/docs/en/stable/api/distributions/generated/pymc.ZeroInflatedNegativeBinomial.html
- Hurdle negative binomial (bundles both hurdle stages into one likelihood): https://www.pymc.io/projects/docs/en/stable/api/distributions/generated/pymc.HurdleNegativeBinomial.html

How the Bayesian (PyMC) approach differs from the frequentist (statsmodels) approach used here:

| | Frequentist (statsmodels) | Bayesian (PyMC) |
|---|---|---|
| Estimation | Maximum likelihood (single point estimate of β) | MCMC sampling from the full posterior p(β \| data) |
| Priors | None (implicit flat prior) | Explicit (e.g. β ~ Normal(0, σ²)), gives built-in regularization |
| Output | Point estimate, standard error, p-value (asymptotic) | Posterior samples, credible interval, full uncertainty over β |
| Inference | Based on large-sample asymptotics | Based on the exact posterior draws |
| Model comparison | AIC / BIC / likelihood ratio | WAIC / LOO / posterior predictive checks |
| Convergence check | Deterministic optimizer (converged or not) | MCMC diagnostics (R-hat, ESS, divergences) |
| Compute cost | Fast (seconds–minutes) | Much slower (minutes–hours) due to MCMC |

In short: the model **structure** (the same ZIP/ZINB/hurdle PMFs and link functions) is identical between the two approaches. The difference is how parameters are estimated and how uncertainty is reported.

## Metrics (planned)

- **RMSE** on validation for comparing predictions.  
- **AIC** for classical models where it applies.  
- **Normalized Gini** — rank-based metric: sort policies by predicted call counts and measure how well actual calls concentrate at the top of the ranking. Range is roughly [-1, 1]; **1.0 = perfect ranking, 0 = random, negative = anti-correlated**.

### Normalized Gini

Sort rows by predicted call counts (descending; highest-risk first). Let N be the number of observations and a<sub>i</sub> the actual call count at rank i (so a₁ is at the highest predicted, a<sub>N</sub> at the lowest predicted). Define:

$$
G(\hat y) = \frac{1}{N}\left(\frac{\sum_{k=1}^{N} \sum_{i=1}^{k} a_i}{\sum_{i=1}^{N} a_i} - \frac{N+1}{2}\right)
$$

Then the normalized Gini is:

$$
\text{Normalized Gini} = \frac{G(\hat y)}{G(y)}
$$

where G(y) is the same quantity computed with predictions equal to the true counts (perfect ranking).

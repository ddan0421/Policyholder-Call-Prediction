# Policyholder Call Prediction

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

- **Hurdle (two stages)** ([Mullahy, 1986](#references); [Gurmu, 1998](#references))  
  - Stage 1: predict P(Y > 0).  
  - Stage 2: on rows with Y > 0, predict E[Y | Y > 0] with a zero-truncated negative binomial.  
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
| Used as P_count in | Poisson GLM, ZIP, hurdle stage 2 (Poisson) | NB regression, ZINB, hurdle stage 2 (zero-truncated NB) |

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

Following [Mullahy (1986)](#references), a hurdle model splits zeros and positives **cleanly**; [Gurmu (1998)](#references) extends this framework to generalized hurdle count regression (e.g. flexible choice of count distribution on the positive part).

1. **Stage 1 (binary)** — logistic regression on Y > 0. All zeros come from this stage only.
2. **Stage 2 (zero-truncated count)** — fit on rows with Y > 0 only, using a count distribution that **cannot produce zeros**.

$$
P(Y=0 \mid X) = \pi(X)
$$

$$
P(Y=k \mid X) = (1-\pi(X)) \cdot \frac{P_{\mathrm{count}}(k \mid X)}{1 - P_{\mathrm{count}}(0 \mid X)}, \quad k > 0
$$

The two stages are estimated **independently** (logistic for stage 1, zero-truncated NB for stage 2 in this project).

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

4. Hurdle: binary + zero-truncated negative binomial ([Mullahy, 1986](#references); [Gurmu, 1998](#references))  
    statsmodels zero-truncated negative binomial: https://www.statsmodels.org/stable/generated/statsmodels.discrete.truncated_model.TruncatedLFNegativeBinomialP.html

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

## Further research

This project intentionally builds the hurdle model "by hand" — fit the stage 1 logistic, fit the stage 2 zero-truncated NB independently, then combine `ŷ = P(Y > 0) × E[Y | Y > 0]` for prediction. The reasons are pedagogical: each stage's coefficients, p-values, and diagnostics stay separately inspectable, and there's no joint-likelihood machinery to debug when something goes wrong.

A few directions that could naturally extend this work but are not demonstrated here:

### All-in-one statsmodels `HurdleCountModel` (not used here)

statsmodels (>= 0.14.0) ships a single-call hurdle implementation that bundles both stages into one `fit()`:

- https://www.statsmodels.org/dev/generated/statsmodels.discrete.truncated_model.HurdleCountModel.html

Sketch:

```python
from statsmodels.discrete.truncated_model import HurdleCountModel

model = HurdleCountModel(
    endog=y,
    exog=X,
    dist="negbin",      # zero-truncated count side ('poisson' or 'negbin')
    zerodist="poisson", # zero hurdle side ('poisson' or 'negbin')
    p=2,                # NB2 parameterization for the count side
)
result = model.fit()
mu_hat = result.predict(X)   # already returns the combined hurdle expectation
```

I can simply do this too, but for this project I am creating the hurdle method by myself so each stage is transparent and the parametric pieces (logit + zero-truncated NB) match the formulas written out above one-for-one. `HurdleCountModel` would have produced a comparable point estimate in a single line — useful as a one-shot baseline if you want to skip the manual two-stage pipeline.

A small caveat for anyone trying it: `HurdleCountModel`'s `zerodist` is not a logistic regression — it uses a Poisson or NegBin model whose `P(Y = 0)` plays the role of the hurdle's "zero probability". That's mathematically a slightly different stage 1 than the logit used in this project. If you want the classical logit-then-NB hurdle, the manual approach in this repo is what you want.

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

### ML hurdle with boosting (not demonstrated here)

Beyond the parametric setup used in this project, the same hurdle decomposition can be implemented with a non-parametric / machine-learning back-end:

- **Stage 1:** a boosted classifier (XGBoost / LightGBM with binary log loss) for P(Y > 0).
- **Stage 2:** a boosted regressor with a count-aware loss (Poisson, zero-truncated Poisson, or a Negative Binomial distribution loss) for E[Y | Y > 0].

This is the natural extension of the parametric hurdle when the data carries strong interactions or non-linearities that a linear log-link cannot capture.

These ML models can be **hard to tune and hard to control**, especially when the positive subset is small (~14K rows for Auto here). They require careful **early stopping** to prevent the booster from memorizing training noise, plus tree-depth / leaf-size regularization, and — for distributional losses like NB — even tighter control because two parameters are predicted per row instead of one. In a quick check with default XGBoost hyperparameters on this dataset, train Gini reached ~0.82 while val Gini collapsed to ~0.09 — the textbook overfit signature. Recovering useful val numbers requires a much larger hyperparameter search than the parametric pipeline, so I stick to the parametric hurdle here.

For readers interested in the ML-hurdle direction, the literature is rich:

- Krasniqi, Bardet, Rynkiewicz (2023). *Parametric and XGBoost Hurdle Model for estimating accident frequency.* HAL preprint: https://hal.science/hal-03739838v2 — uses XGBoost for stage 2 with a custom zero-truncated Poisson loss, applied to a French car-insurance portfolio.
- Xu, Ye, Gao, Chu (2024). *Generalized hurdle count data models based on interpretable machine learning with an application to health care demand.* *Computing* 106:295–325: https://doi.org/10.1007/s00607-023-01224-3 — extends the hurdle to decision tree / random forest / SVM / XGBoost in both stages, with variable importance and break-down plots for interpretability.

## References

- Mullahy, J. (1986). Specification and testing of some modified count data models. *Journal of Econometrics*, 33, 341–365. https://doi.org/10.1016/0304-4076(86)90002-3

- Gurmu, S. (1998). Generalized hurdle count data regression models. *Economics Letters*, 58(3), 263–268. https://doi.org/10.1016/S0165-1765(97)00295-4

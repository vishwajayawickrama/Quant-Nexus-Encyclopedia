# Module 33: Regime Detection & Adaptive Strategies

**Prerequisites:** Modules 02 (Probability & Measure Theory), 21 (Time Series), 22 (Kalman Filters), 26 (Machine Learning)
**Builds toward:** Module 34 (Alternative Data & Feature Engineering)

---

## Table of Contents

1. [Hidden Markov Models](#1-hidden-markov-models)
2. [Forward-Backward Algorithm](#2-forward-backward-algorithm)
3. [Viterbi Decoding](#3-viterbi-decoding)
4. [Baum-Welch EM Learning](#4-baum-welch-em-learning)
5. [Model Selection & Regime Interpretation](#5-model-selection--regime-interpretation)
6. [Change-Point Detection: Offline Methods](#6-change-point-detection-offline-methods)
7. [Change-Point Detection: Online Methods](#7-change-point-detection-online-methods)
8. [Structural Break Tests](#8-structural-break-tests)
9. [Markov-Switching Models](#9-markov-switching-models)
10. [Spectral & Wavelet-Based Regime Detection](#10-spectral--wavelet-based-regime-detection)
11. [Ensemble Regime Detection](#11-ensemble-regime-detection)
12. [Transition Dynamics & Regime Persistence](#12-transition-dynamics--regime-persistence)
13. [Adaptive Strategies](#13-adaptive-strategies)
14. [Implementation: Python](#14-implementation-python)
15. [Implementation: C++](#15-implementation-c)
16. [Exercises](#16-exercises)
17. [Summary and Concept Map](#17-summary-and-concept-map)

---

## 1. Hidden Markov Models

Financial markets exhibit behavior that shifts between qualitatively distinct regimes --- bull markets with steady appreciation and low volatility, bear markets with sharp declines and volatility clustering, and range-bound markets with mean-reverting dynamics. The Hidden Markov Model (HMM) provides a principled probabilistic framework for modeling these latent regimes and inferring which regime governs the market at any point in time.

### 1.1 Model Specification

An HMM is defined by the tuple $\lambda = (\boldsymbol{\pi}, \mathbf{A}, \boldsymbol{\theta})$ where:

**Hidden states.** A discrete random variable $S_t \in \{1, 2, \ldots, K\}$ evolves as a first-order Markov chain. The number of states $K$ is fixed (typically $K = 2$ or $K = 3$ for financial applications).

**Transition matrix.** The $K \times K$ matrix $\mathbf{A}$ has entries:

$$a_{ij} = P(S_t = j \mid S_{t-1} = i), \quad \sum_{j=1}^{K} a_{ij} = 1 \;\; \forall \, i$$

Each row of $\mathbf{A}$ is a probability distribution over next states. High diagonal entries $a_{ii}$ indicate persistent regimes.

**Initial distribution.** The row vector $\boldsymbol{\pi} = (\pi_1, \ldots, \pi_K)$ specifies $\pi_k = P(S_1 = k)$.

**Emission distributions.** Conditional on $S_t = k$, the observation $y_t$ is drawn from an emission distribution $f(y_t \mid \boldsymbol{\theta}_k)$. For financial returns, the most common choices are:

| Emission model | $f(y_t \mid S_t = k)$ | Use case |
|---|---|---|
| Gaussian | $\mathcal{N}(\mu_k, \sigma_k^2)$ | Basic bull/bear detection |
| Student-$t$ | $t_{\nu_k}(\mu_k, \sigma_k^2)$ | Fat tails within regimes |
| Gaussian mixture | $\sum_m w_{km} \mathcal{N}(\mu_{km}, \sigma_{km}^2)$ | Multimodal within-regime returns |
| Multivariate Gaussian | $\mathcal{N}(\boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)$ | Joint modeling of multiple assets |

The parameter vector $\boldsymbol{\theta} = \{\boldsymbol{\theta}_1, \ldots, \boldsymbol{\theta}_K\}$ collects all emission parameters. For the Gaussian case, $\boldsymbol{\theta}_k = (\mu_k, \sigma_k^2)$.

### 1.2 Conditional Independence Structure

The HMM encodes two key conditional independence assumptions:

1. **Markov property of states:** $P(S_t \mid S_{t-1}, S_{t-2}, \ldots, S_1) = P(S_t \mid S_{t-1})$
2. **Output independence:** $P(y_t \mid S_1, \ldots, S_T, y_1, \ldots, y_T) = P(y_t \mid S_t)$

The joint probability of a state sequence $\mathbf{s} = (s_1, \ldots, s_T)$ and observation sequence $\mathbf{y} = (y_1, \ldots, y_T)$ factorizes as:

$$P(\mathbf{y}, \mathbf{s} \mid \lambda) = \pi_{s_1} \prod_{t=2}^{T} a_{s_{t-1}, s_t} \prod_{t=1}^{T} f(y_t \mid \boldsymbol{\theta}_{s_t})$$

This factorization structure enables efficient inference algorithms with $O(TK^2)$ complexity rather than the naive $O(K^T)$ enumeration.

### 1.3 The Three Canonical HMM Problems

| Problem | Question | Algorithm |
|---|---|---|
| Evaluation | $P(\mathbf{y} \mid \lambda)$? | Forward algorithm |
| Decoding | $\arg\max_\mathbf{s} P(\mathbf{s} \mid \mathbf{y}, \lambda)$? | Viterbi |
| Learning | $\arg\max_\lambda P(\mathbf{y} \mid \lambda)$? | Baum-Welch (EM) |

---

## 2. Forward-Backward Algorithm

The forward-backward algorithm computes two quantities that together enable all posterior inference about hidden states.

### 2.1 Forward Variables (Alpha Recursion)

Define the **forward variable**:

$$\alpha_t(j) = P(y_1, y_2, \ldots, y_t, S_t = j \mid \lambda)$$

This is the joint probability of observing the partial sequence $y_1, \ldots, y_t$ and being in state $j$ at time $t$.

**Initialization ($t = 1$):**

$$\alpha_1(j) = \pi_j \, f(y_1 \mid \boldsymbol{\theta}_j), \quad j = 1, \ldots, K$$

**Recursion ($t = 2, \ldots, T$):** To arrive at state $j$ at time $t$, we must have been in some state $i$ at time $t-1$ and transitioned to $j$:

$$\alpha_t(j) = \left[\sum_{i=1}^{K} \alpha_{t-1}(i) \, a_{ij}\right] f(y_t \mid \boldsymbol{\theta}_j)$$

**Derivation.** By the law of total probability and the Markov/output independence properties:

$$\alpha_t(j) = P(y_{1:t}, S_t = j) = \sum_{i=1}^K P(y_{1:t-1}, S_{t-1} = i) \cdot P(S_t = j \mid S_{t-1} = i) \cdot P(y_t \mid S_t = j)$$

$$= \sum_{i=1}^K \alpha_{t-1}(i) \cdot a_{ij} \cdot f(y_t \mid \boldsymbol{\theta}_j)$$

**Likelihood.** The total data likelihood is:

$$P(\mathbf{y} \mid \lambda) = \sum_{j=1}^K \alpha_T(j)$$

**Numerical stability.** Direct computation of $\alpha_t(j)$ underflows to zero for large $T$ because it is a product of many probabilities. The standard remedy is to use **scaled forward variables**:

$$\hat{\alpha}_t(j) = \frac{\alpha_t(j)}{c_t}, \quad c_t = \sum_{j=1}^{K} \alpha_t(j)$$

so that $\sum_j \hat{\alpha}_t(j) = 1$ at each time step. The log-likelihood is then $\log P(\mathbf{y} \mid \lambda) = \sum_{t=1}^T \log c_t$.

### 2.2 Backward Variables (Beta Recursion)

Define the **backward variable**:

$$\beta_t(i) = P(y_{t+1}, y_{t+2}, \ldots, y_T \mid S_t = i, \lambda)$$

This is the probability of future observations given the current state.

**Initialization ($t = T$):**

$$\beta_T(i) = 1, \quad i = 1, \ldots, K$$

**Recursion ($t = T-1, \ldots, 1$):**

$$\beta_t(i) = \sum_{j=1}^{K} a_{ij} \, f(y_{t+1} \mid \boldsymbol{\theta}_j) \, \beta_{t+1}(j)$$

**Derivation.** Marginalizing over the next state:

$$\beta_t(i) = P(y_{t+1:T} \mid S_t = i) = \sum_{j=1}^K P(S_{t+1} = j \mid S_t = i) \cdot P(y_{t+1} \mid S_{t+1} = j) \cdot P(y_{t+2:T} \mid S_{t+1} = j)$$

$$= \sum_{j=1}^K a_{ij} \cdot f(y_{t+1} \mid \boldsymbol{\theta}_j) \cdot \beta_{t+1}(j)$$

### 2.3 Posterior State Probabilities

Combining forward and backward variables yields the **smoothed state posterior**:

$$\gamma_t(j) = P(S_t = j \mid \mathbf{y}, \lambda) = \frac{\alpha_t(j) \, \beta_t(j)}{P(\mathbf{y} \mid \lambda)} = \frac{\alpha_t(j) \, \beta_t(j)}{\sum_{k=1}^{K} \alpha_t(k) \, \beta_t(k)}$$

This is the probability that the market was in regime $j$ at time $t$, given the *entire* observation history (both past and future). For real-time trading, only the **filtered probability** $P(S_t = j \mid y_{1:t})$ is available:

$$P(S_t = j \mid y_{1:t}) = \frac{\alpha_t(j)}{\sum_{k=1}^K \alpha_t(k)}$$

The **transition posterior** is also needed for learning:

$$\xi_t(i, j) = P(S_t = i, S_{t+1} = j \mid \mathbf{y}, \lambda) = \frac{\alpha_t(i) \, a_{ij} \, f(y_{t+1} \mid \boldsymbol{\theta}_j) \, \beta_{t+1}(j)}{P(\mathbf{y} \mid \lambda)}$$

---

## 3. Viterbi Decoding

The Viterbi algorithm finds the single most probable state sequence:

$$\mathbf{s}^* = \arg\max_{s_1, \ldots, s_T} P(S_1 = s_1, \ldots, S_T = s_T \mid \mathbf{y}, \lambda)$$

### 3.1 Dynamic Programming Derivation

Define:

$$\delta_t(j) = \max_{s_1, \ldots, s_{t-1}} P(S_1 = s_1, \ldots, S_{t-1} = s_{t-1}, S_t = j, y_1, \ldots, y_t \mid \lambda)$$

This is the probability of the best partial path ending in state $j$ at time $t$.

**Initialization:**

$$\delta_1(j) = \pi_j \, f(y_1 \mid \boldsymbol{\theta}_j), \quad \psi_1(j) = 0$$

**Recursion:** The key insight is the **principle of optimality** --- the best path to state $j$ at time $t$ must pass through the best path to some state $i$ at time $t-1$:

$$\delta_t(j) = \max_{i} \left[\delta_{t-1}(i) \, a_{ij}\right] \cdot f(y_t \mid \boldsymbol{\theta}_j)$$

$$\psi_t(j) = \arg\max_{i} \left[\delta_{t-1}(i) \, a_{ij}\right]$$

where $\psi_t(j)$ stores the backpointer --- which previous state maximized the path probability.

**Termination:**

$$s_T^* = \arg\max_j \delta_T(j)$$

**Backtracking:** For $t = T-1, T-2, \ldots, 1$:

$$s_t^* = \psi_{t+1}(s_{t+1}^*)$$

**Complexity.** Time $O(TK^2)$, space $O(TK)$. In practice, we work in log space to avoid underflow: $\log\delta_t(j) = \max_i [\log\delta_{t-1}(i) + \log a_{ij}] + \log f(y_t \mid \boldsymbol{\theta}_j)$.

### 3.2 Viterbi vs. Pointwise MAP

The Viterbi path $\mathbf{s}^*$ maximizes the joint probability of the entire state sequence. An alternative is the **pointwise MAP** path: $\hat{s}_t = \arg\max_j \gamma_t(j)$. These can differ: the pointwise MAP may select transitions with $a_{ij} = 0$, producing an impossible sequence. Viterbi guarantees a valid path but may assign a state at time $t$ that is not the most likely marginal state. For trading, Viterbi is preferred when you need a coherent regime narrative; pointwise MAP is preferred when you care only about the current regime probability.

---

## 4. Baum-Welch EM Learning

The Baum-Welch algorithm is a special case of Expectation-Maximization (EM) applied to HMMs. It iteratively maximizes the data likelihood $P(\mathbf{y} \mid \lambda)$.

### 4.1 E-Step

Run the forward-backward algorithm to compute $\gamma_t(j)$ and $\xi_t(i,j)$ for all $t, i, j$ using the current parameter estimates $\lambda^{(n)}$.

### 4.2 M-Step

Update the parameters by maximizing the expected complete-data log-likelihood:

$$Q(\lambda, \lambda^{(n)}) = \mathbb{E}_{\mathbf{S} \mid \mathbf{y}, \lambda^{(n)}} \left[\log P(\mathbf{y}, \mathbf{S} \mid \lambda)\right]$$

**Initial distribution:**

$$\pi_k^{(n+1)} = \gamma_1(k)$$

**Transition matrix:**

$$a_{ij}^{(n+1)} = \frac{\sum_{t=1}^{T-1} \xi_t(i,j)}{\sum_{t=1}^{T-1} \gamma_t(i)}$$

**Derivation.** The expected complete-data log-likelihood for the transition parameters is:

$$Q_A = \sum_{t=1}^{T-1} \sum_{i=1}^{K} \sum_{j=1}^{K} \xi_t(i,j) \log a_{ij}$$

Maximizing $Q_A$ subject to $\sum_j a_{ij} = 1$ using a Lagrange multiplier gives:

$$\mathcal{L} = Q_A - \sum_i \lambda_i \left(\sum_j a_{ij} - 1\right)$$

$$\frac{\partial \mathcal{L}}{\partial a_{ij}} = \frac{\sum_t \xi_t(i,j)}{a_{ij}} - \lambda_i = 0 \implies a_{ij} \propto \sum_t \xi_t(i,j)$$

Normalizing to sum to one over $j$ yields the update above.

**Gaussian emission parameters:** For $f(y_t \mid \boldsymbol{\theta}_k) = \mathcal{N}(y_t; \mu_k, \sigma_k^2)$:

$$\mu_k^{(n+1)} = \frac{\sum_{t=1}^{T} \gamma_t(k) \, y_t}{\sum_{t=1}^{T} \gamma_t(k)}$$

$$(\sigma_k^2)^{(n+1)} = \frac{\sum_{t=1}^{T} \gamma_t(k) \, (y_t - \mu_k^{(n+1)})^2}{\sum_{t=1}^{T} \gamma_t(k)}$$

These are the responsibility-weighted mean and variance, exactly analogous to Gaussian mixture EM.

### 4.3 Convergence Properties

Baum-Welch inherits the EM guarantee that $P(\mathbf{y} \mid \lambda^{(n+1)}) \geq P(\mathbf{y} \mid \lambda^{(n)})$, converging to a local maximum. Practical considerations:

- **Multiple restarts:** Run 10--50 random initializations and keep the best.
- **Initialization heuristics:** K-means on returns to seed emission parameters; uniform transition matrix with slight diagonal dominance.
- **Convergence criterion:** $|\log P(\mathbf{y} \mid \lambda^{(n+1)}) - \log P(\mathbf{y} \mid \lambda^{(n)})| < \epsilon$ with $\epsilon \sim 10^{-6}$.

---

## 5. Model Selection & Regime Interpretation

### 5.1 Choosing the Number of States

The likelihood increases monotonically with $K$, so information criteria penalize complexity:

**AIC (Akaike):**

$$\text{AIC} = -2 \log P(\mathbf{y} \mid \hat{\lambda}) + 2p$$

**BIC (Bayesian):**

$$\text{BIC} = -2 \log P(\mathbf{y} \mid \hat{\lambda}) + p \log T$$

where the number of free parameters for a $K$-state Gaussian HMM is:

$$p = (K - 1) + K(K - 1) + 2K = K^2 + 2K - 1$$

The first term counts the initial distribution ($K - 1$ free probabilities), the second counts transition parameters ($K$ rows, each with $K - 1$ free parameters), and the third counts the $2K$ Gaussian parameters ($\mu_k, \sigma_k^2$ for each state).

BIC is preferred in practice because it imposes a stronger penalty for large $T$ (typical in finance), reducing overfitting. Cross-validated log-likelihood on held-out data is the most reliable approach when computational budget allows.

### 5.2 Regime Interpretation in Finance

For a $K = 2$ model fitted to equity returns, the states typically correspond to:

| Property | State 1 ("Bull") | State 2 ("Bear") |
|---|---|---|
| Mean return $\mu_k$ | Positive, moderate | Negative or near zero |
| Volatility $\sigma_k$ | Low (10--15% ann.) | High (25--40% ann.) |
| Persistence $a_{kk}$ | 0.95--0.99 | 0.90--0.97 |
| Avg. duration $1/(1-a_{kk})$ | 20--100 days | 10--30 days |

For $K = 3$, a third "crisis" or "recovery" state often emerges with very high volatility and/or strong positive mean (snap-back rallies).

---

## 6. Change-Point Detection: Offline Methods

While HMMs model recurring regimes with transitions, change-point detection identifies one-time structural breaks.

### 6.1 Binary Segmentation

The simplest approach: find the single change point that most improves the fit, then recurse on each segment.

1. For each candidate split point $\tau \in \{1, \ldots, T-1\}$, compute the cost reduction: $\Delta C(\tau) = C(y_{1:T}) - [C(y_{1:\tau}) + C(y_{\tau+1:T})]$
2. Select $\tau^* = \arg\max_\tau \Delta C(\tau)$
3. If $\Delta C(\tau^*) > \beta$ (penalty threshold), accept the split and recurse on each segment

Binary segmentation is $O(T \log T)$ but is **greedy** and can miss change points that are only evident jointly.

### 6.2 PELT: Pruned Exact Linear Time

PELT (Killick et al., 2012) finds the exact optimal segmentation in expected $O(T)$ time.

**Objective.** Minimize the penalized cost:

$$\sum_{i=0}^{m} \left[C(y_{\tau_i+1:\tau_{i+1}}) + \beta\right]$$

over all segmentations $0 = \tau_0 < \tau_1 < \cdots < \tau_m < \tau_{m+1} = T$, where $C(\cdot)$ is a segment cost (e.g., negative Gaussian log-likelihood) and $\beta$ is the penalty per change point.

**Dynamic programming.** Define:

$$F(t) = \min_{\tau_1, \ldots, \tau_m} \left\{\sum_{i=0}^{m} \left[C(y_{\tau_i+1:\tau_{i+1}}) + \beta\right]\right\} \quad \text{over segmentations of } y_{1:t}$$

Then:

$$F(t) = \min_{0 \leq s < t} \left[F(s) + C(y_{s+1:t}) + \beta\right]$$

with $F(0) = -\beta$. The naive DP is $O(T^2)$.

**Pruning condition (PELT).** The key insight is that candidate split points can be pruned. If $C$ satisfies the inequality:

$$C(y_{s+1:t}) + C(y_{t+1:T}) + \beta \leq C(y_{s+1:T})$$

for some constant (which holds for costs in the exponential family), then at time $t$, any candidate $s$ satisfying:

$$F(s) + C(y_{s+1:t}) \geq F(t)$$

can never be optimal for any future time $t' > t$ and is pruned from the candidate set. Under the assumption that change points are spread across the data, the expected number of candidates at each step is bounded, yielding $O(T)$ expected complexity.

### 6.3 BIC Penalty Calibration

The penalty $\beta$ controls the trade-off between fit and complexity. Common choices:

- **BIC:** $\beta = p \log T / 2$ where $p$ is the number of parameters per segment
- **mBIC (modified BIC):** adds $\log \binom{T}{m}$ term for the number of ways to place $m$ change points
- **CROPS:** Computes the optimal segmentation for a range of $\beta$ values simultaneously, allowing the analyst to inspect an "elbow" plot

---

## 7. Change-Point Detection: Online Methods

### 7.1 CUSUM Charts

The Cumulative Sum (CUSUM) chart detects shifts in the mean of a process. Define:

$$S_t^+ = \max(0, S_{t-1}^+ + (y_t - \mu_0) - k), \quad S_0^+ = 0$$

$$S_t^- = \max(0, S_{t-1}^- - (y_t - \mu_0) - k), \quad S_0^- = 0$$

An alarm is raised when $S_t^+$ or $S_t^-$ exceeds a threshold $h$. The **allowance** $k$ is typically set to half the shift to detect: $k = |\mu_1 - \mu_0|/2$.

### 7.2 BOCPD: Bayesian Online Change Point Detection

Adams & MacKay (2007) introduced BOCPD, which maintains a posterior over the **run length** $r_t$ --- the number of time steps since the last change point.

**Run length posterior recursion.** Let $r_t$ denote the run length at time $t$. The key derivation proceeds as follows.

The joint distribution of run length and data factorizes as:

$$P(r_t, y_{1:t}) = \sum_{r_{t-1}} P(r_t \mid r_{t-1}) \, P(y_t \mid r_{t-1}, y_t^{(r)}) \, P(r_{t-1}, y_{1:t-1})$$

where $y_t^{(r)}$ denotes the observations in the current run.

The **changepoint prior** specifies the transition:

$$P(r_t = r_{t-1} + 1 \mid r_{t-1}) = 1 - H(r_{t-1}) \quad \text{(growth)}$$

$$P(r_t = 0 \mid r_{t-1}) = H(r_{t-1}) \quad \text{(change point)}$$

where $H(\tau)$ is the **hazard function** --- the probability that a run of length $\tau$ ends. For a geometric prior on run length, $H(\tau) = 1/\lambda$ is constant (memoryless), where $\lambda$ is the expected run length.

**Growth probabilities (run continues):**

$$P(r_t = r_{t-1} + 1, y_{1:t}) = P(y_t \mid r_{t-1}, y_t^{(r)}) \cdot (1 - H(r_{t-1})) \cdot P(r_{t-1}, y_{1:t-1})$$

**Changepoint probability (run resets):**

$$P(r_t = 0, y_{1:t}) = P(y_t \mid r_t = 0) \sum_{r_{t-1}} H(r_{t-1}) \cdot P(r_{t-1}, y_{1:t-1})$$

**Predictive likelihood.** For conjugate-exponential models, $P(y_t \mid r_{t-1}, y_t^{(r)})$ has a closed form. For a Gaussian with unknown mean and known variance $\sigma^2$, with prior $\mu \sim \mathcal{N}(\mu_0, \sigma_0^2)$:

$$P(y_t \mid r_{t-1}, y_t^{(r)}) = \mathcal{N}\left(y_t; \bar{\mu}_{r}, \sigma^2 + \sigma_r^2\right)$$

where $\bar{\mu}_r$ and $\sigma_r^2$ are the posterior mean and variance of $\mu$ given the run's data.

The posterior run-length distribution $P(r_t \mid y_{1:t})$ is obtained by normalizing:

$$P(r_t \mid y_{1:t}) = \frac{P(r_t, y_{1:t})}{\sum_{r} P(r_t = r, y_{1:t})}$$

A change point is detected when $P(r_t = 0 \mid y_{1:t})$ exceeds a threshold.

---

## 8. Structural Break Tests

### 8.1 Chow Test and Its Limitations

The classical Chow test for a single known break date $\tau$ in a linear regression $y_t = \mathbf{x}_t^\top \boldsymbol{\beta} + \varepsilon_t$ tests $H_0: \boldsymbol{\beta}_1 = \boldsymbol{\beta}_2$ using an $F$-statistic. However, it requires the break date to be known a priori.

### 8.2 Bai-Perron Test

The Bai-Perron (1998, 2003) framework tests for multiple structural breaks at unknown dates in linear regression models.

**Sup-$F$ test.** For a single break at unknown date $\tau$:

$$\text{sup}F_T = \sup_{\tau \in [\epsilon T, (1-\epsilon)T]} F_T(\tau)$$

where $F_T(\tau)$ is the Chow $F$-statistic for a break at $\tau$ and $\epsilon$ is a trimming parameter (typically 0.15) that excludes endpoints. The distribution under $H_0$ is non-standard and tabulated by Andrews (1993).

**Sequential detection of $m$ breaks.** The Bai-Perron procedure:

1. Test $H_0: 0$ breaks vs $H_1: m$ breaks using $\text{sup}F_T(m)$
2. Use the **sequential** test: given $\ell$ breaks, test for $\ell + 1$ vs $\ell$
3. Estimate break dates by global minimization of sum of squared residuals using dynamic programming in $O(T^2)$

### 8.3 Quandt-Andrews Test

The Quandt-Andrews (also called Andrews' sup-Wald) test is closely related. It computes:

$$\text{sup}W = \sup_{\tau \in [\underline{\tau}, \bar{\tau}]} W_T(\tau)$$

where $W_T(\tau)$ is the Wald statistic. The $p$-values use Hansen's (1997) approximate distribution. This test is available in most econometrics packages and is the go-to first test for an unknown single break.

---

## 9. Markov-Switching Models

### 9.1 Hamilton's Regime-Switching Model

Hamilton (1989) introduced the Markov-switching autoregression, the foundational model in regime-switching econometrics. The model specifies:

$$y_t = \mu_{S_t} + \phi_1 (y_{t-1} - \mu_{S_{t-1}}) + \varepsilon_t, \quad \varepsilon_t \sim \mathcal{N}(0, \sigma_{S_t}^2)$$

where $S_t$ follows a Markov chain with transition matrix $\mathbf{A}$.

**Filter probability recursion.** The Hamilton filter computes the **filtered probability** $P(S_t = j \mid y_{1:t}; \lambda)$ recursively.

**Step 1 --- Prediction.** Given the filtered probability at $t-1$:

$$P(S_t = j \mid y_{1:t-1}) = \sum_{i=1}^K a_{ij} \, P(S_{t-1} = i \mid y_{1:t-1})$$

This is a matrix-vector multiplication: $\boldsymbol{\xi}_{t|t-1} = \mathbf{A}^\top \boldsymbol{\xi}_{t-1|t-1}$.

**Step 2 --- Update.** Upon observing $y_t$:

$$P(S_t = j \mid y_{1:t}) = \frac{f(y_t \mid S_t = j, y_{1:t-1}) \, P(S_t = j \mid y_{1:t-1})}{\sum_{k=1}^K f(y_t \mid S_t = k, y_{1:t-1}) \, P(S_t = k \mid y_{1:t-1})}$$

**Derivation.** This is a direct application of Bayes' theorem:

$$P(S_t = j \mid y_{1:t}) = P(S_t = j \mid y_t, y_{1:t-1}) = \frac{P(y_t \mid S_t = j, y_{1:t-1}) P(S_t = j \mid y_{1:t-1})}{P(y_t \mid y_{1:t-1})}$$

The denominator is the marginal predictive density, which serves as the normalizing constant. The log-likelihood is:

$$\log \mathcal{L}(\lambda) = \sum_{t=1}^T \log P(y_t \mid y_{1:t-1}) = \sum_{t=1}^T \log \left[\sum_{k=1}^K f(y_t \mid S_t = k, y_{1:t-1}) \, P(S_t = k \mid y_{1:t-1})\right]$$

This is maximized numerically (e.g., by EM or gradient-based methods).

### 9.2 MS-GARCH

Combining Markov switching with GARCH captures both regime shifts in volatility dynamics and within-regime volatility clustering:

$$y_t = \mu_{S_t} + \varepsilon_t, \quad \varepsilon_t = \sigma_t z_t, \quad z_t \sim \mathcal{N}(0,1)$$

$$\sigma_t^2 = \omega_{S_t} + \alpha_{S_t} \varepsilon_{t-1}^2 + \beta_{S_t} \sigma_{t-1}^2$$

The challenge is path dependence: $\sigma_t^2$ depends on the entire history of regimes, not just the current state. Approximations include:

- **Collapsing:** At each step, collapse $K$ conditional variances to a single weighted variance (Klaassen, 2002)
- **Viterbi path:** Condition on the most likely state sequence
- **Particle MCMC:** Full Bayesian inference via sequential Monte Carlo

---

## 10. Spectral & Wavelet-Based Regime Detection

### 10.1 Wavelet-Based Regime Detection

Wavelets provide a time-frequency decomposition that naturally captures non-stationary behavior.

**Discrete Wavelet Transform (DWT).** The DWT decomposes a signal into detail coefficients $d_{j,k}$ at scale $j$ and translation $k$:

$$d_{j,k} = \sum_t y_t \, \psi_{j,k}(t), \quad \psi_{j,k}(t) = 2^{-j/2} \psi(2^{-j}t - k)$$

**Regime detection via wavelet variance.** Compute the wavelet variance at each scale:

$$\hat{\nu}_j^2 = \frac{1}{N_j} \sum_k d_{j,k}^2$$

A sudden increase in $\hat{\nu}_j^2$ at coarse scales signals a regime change.

**MODWT (Maximal Overlap DWT)** is preferred over the decimated DWT because it is shift-invariant and aligns wavelet coefficients with the original time series.

### 10.2 Hilbert-Huang Transform

The Hilbert-Huang Transform (HHT) is a purely data-driven decomposition for non-stationary, non-linear signals.

**Step 1: Empirical Mode Decomposition (EMD).** Decompose the signal into **Intrinsic Mode Functions** (IMFs) $c_j(t)$ via sifting:

1. Identify local maxima and minima of $y(t)$
2. Compute upper/lower envelopes by cubic spline interpolation
3. Subtract the mean envelope: $h(t) = y(t) - \text{mean envelope}$
4. Repeat until $h(t)$ satisfies the IMF criteria (zero crossings $\approx$ extrema count)
5. Set $c_1(t) = h(t)$, compute the residual $r(t) = y(t) - c_1(t)$, and iterate on $r(t)$

**Step 2: Hilbert Spectrum.** Apply the Hilbert transform to each IMF to obtain instantaneous frequency and amplitude:

$$\tilde{c}_j(t) = \frac{1}{\pi} \text{P.V.} \int_{-\infty}^{\infty} \frac{c_j(\tau)}{t - \tau} d\tau$$

The analytic signal $z_j(t) = c_j(t) + i\tilde{c}_j(t) = A_j(t) e^{i\theta_j(t)}$ yields instantaneous frequency $\omega_j(t) = d\theta_j/dt$ and amplitude $A_j(t)$.

**Financial application.** Regime changes appear as abrupt shifts in the instantaneous frequency of dominant IMFs, particularly the low-frequency components that capture trend behavior.

---

## 11. Ensemble Regime Detection

Single regime detectors are noisy. Combining multiple indicators produces more robust regime classification.

### 11.1 Multi-Indicator Voting

Define $K$ binary regime classifiers $\hat{S}_t^{(k)} \in \{0, 1\}$ (bear/bull):

$$\hat{S}_t^{\text{ensemble}} = \mathbb{1}\left[\sum_{k=1}^{K} w_k \hat{S}_t^{(k)} > 0.5\right]$$

Candidate classifiers include: HMM filtered probability, BOCPD run length, 200-day moving average crossover, VIX regime (above/below 20), yield curve slope sign, credit spread level, momentum breadth.

### 11.2 Stacking

A meta-learner (logistic regression, gradient boosted tree) takes the outputs of base regime detectors as features and learns to predict the true regime label. The challenge is defining "ground truth" --- one approach uses the smoothed HMM posterior on an extended lookback window as a noisy label.

### 11.3 Regime Probability Aggregation

Instead of hard labels, aggregate soft probabilities. Given $M$ models producing regime probability estimates $p_m(t) = P_m(S_t = \text{bull})$:

$$\bar{p}(t) = \frac{\sum_{m=1}^M w_m \, p_m(t)}{\sum_{m=1}^M w_m}$$

Weights $w_m$ can be calibrated by inverse Brier score on a validation set:

$$\text{Brier}_m = \frac{1}{T_{\text{val}}} \sum_{t=1}^{T_{\text{val}}} (p_m(t) - S_t^{\text{true}})^2, \quad w_m \propto \frac{1}{\text{Brier}_m}$$

---

## 12. Transition Dynamics & Regime Persistence

### 12.1 Expected Regime Duration

For a Markov chain with self-transition probability $a_{ii}$, the sojourn time in state $i$ is geometrically distributed:

$$P(D_i = d) = a_{ii}^{d-1}(1 - a_{ii}), \quad \mathbb{E}[D_i] = \frac{1}{1 - a_{ii}}$$

For $a_{ii} = 0.98$, the expected duration is 50 time steps. The geometric distribution imposes a memoryless property: the probability of leaving the regime is constant regardless of how long you have been in it.

### 12.2 Semi-Markov Extensions

**Hidden Semi-Markov Models (HSMMs)** replace the geometric sojourn distribution with an explicit duration distribution $P(D_i = d)$, allowing more realistic persistence patterns:

- **Negative binomial:** allows duration clustering
- **Log-normal:** fat-tailed durations (long bull markets)
- **Empirical:** nonparametric, estimated from historical regime durations

The HSMM modifies the forward recursion to track both state and remaining duration, increasing complexity to $O(TK^2 D_{\max})$.

### 12.3 Predicting Regime Transitions

Features that historically precede regime transitions include:

| Feature | Signal |
|---|---|
| Yield curve inversion | Bear market warning (12--18 month lead) |
| Credit spread widening | Stress regime imminent |
| VIX term structure backwardation | Market in crisis |
| Declining market breadth | Bull market exhaustion |
| Rising default rates | Credit regime deterioration |

A logistic regression or survival model on these features can estimate the probability of transition out of the current regime:

$$P(\text{transition at } t+1 \mid \mathbf{x}_t) = \sigma(\boldsymbol{\beta}^\top \mathbf{x}_t)$$

---

## 13. Adaptive Strategies

### 13.1 Regime-Conditional Portfolio Allocation

Given regime probabilities $\boldsymbol{\gamma}_t = (\gamma_t(1), \ldots, \gamma_t(K))^\top$, define regime-conditional optimal portfolios $\mathbf{w}_k^*$ and blend:

$$\mathbf{w}_t = \sum_{k=1}^{K} \gamma_t(k) \, \mathbf{w}_k^*$$

For a two-state model:
- **Bull ($k=1$):** Maximize Sharpe ratio with full equity exposure
- **Bear ($k=2$):** Minimize drawdown with bonds/cash tilt

The blended portfolio transitions smoothly between extreme allocations as regime probabilities shift.

### 13.2 Adaptive Kelly Criterion

The Kelly criterion (Module 24) prescribes optimal bet sizing under known distribution parameters. Under regime uncertainty, the **adaptive Kelly** adjusts:

$$f_t^* = \sum_{k=1}^{K} \gamma_t(k) \, f_k^*$$

where $f_k^*$ is the Kelly fraction for regime $k$:

$$f_k^* = \frac{\mu_k}{\sigma_k^2}$$

**Regime-uncertainty penalty.** Because regime misclassification risk inflates realized variance, a conservative approach scales down:

$$f_t^{\text{adj}} = f_t^* \cdot \left(1 - H(\boldsymbol{\gamma}_t)\right)$$

where $H(\boldsymbol{\gamma}_t) = -\sum_k \gamma_t(k) \log \gamma_t(k) / \log K$ is the normalized entropy of the regime distribution. When the regime is certain ($H = 0$), full Kelly is used. When maximally uncertain ($H = 1$), the position is zero.

### 13.3 Exponentially Weighted Online Learning

Rather than discrete regime labels, an **adaptive forgetting factor** continuously adjusts the effective sample size:

$$\hat{\mu}_t = (1 - \lambda_t) \hat{\mu}_{t-1} + \lambda_t y_t$$

$$\hat{\sigma}_t^2 = (1 - \lambda_t) \hat{\sigma}_{t-1}^2 + \lambda_t (y_t - \hat{\mu}_t)^2$$

The forgetting factor $\lambda_t$ is made adaptive by linking it to the BOCPD change-point probability:

$$\lambda_t = \lambda_{\min} + (\lambda_{\max} - \lambda_{\min}) \cdot P(r_t = 0 \mid y_{1:t})$$

When a change point is likely, $\lambda_t$ increases (shorter memory); during stable periods, $\lambda_t$ decreases (longer memory).

### 13.4 Meta-Strategy Selection

Given a library of strategies $\{S_1, \ldots, S_M\}$ with historical performance conditional on regime, select or blend strategies based on the current regime:

$$R_t^{\text{meta}} = \sum_{m=1}^{M} w_{m,t} \, R_t^{(m)}, \quad w_{m,t} \propto \exp\left(\eta \sum_{k=1}^K \gamma_t(k) \, \bar{R}_k^{(m)}\right)$$

where $\bar{R}_k^{(m)}$ is the historical performance of strategy $m$ in regime $k$ and $\eta$ is a temperature parameter. This is a softmax allocation that tilts toward strategies suited to the likely current regime.

---

## 14. Implementation: Python

```python
"""
Module 33: Regime Detection & Adaptive Strategies — Python Implementation
Requires: numpy, scipy, numba
"""

import numpy as np
from scipy.special import logsumexp
from scipy.stats import norm
from numba import njit
from typing import NamedTuple


# ============================================================================
# 1. Hidden Markov Model — Full Implementation
# ============================================================================

class HMMParams(NamedTuple):
    """Parameters for a Gaussian HMM."""
    pi: np.ndarray       # (K,) initial distribution
    A: np.ndarray        # (K, K) transition matrix
    mu: np.ndarray       # (K,) emission means
    sigma: np.ndarray    # (K,) emission standard deviations


@njit
def _forward_scaled(y: np.ndarray, pi: np.ndarray, A: np.ndarray,
                    mu: np.ndarray, sigma: np.ndarray):
    """
    Scaled forward algorithm. Returns:
        alpha_hat: (T, K) scaled forward variables
        c: (T,) scaling factors
        log_lik: scalar log-likelihood
    """
    T = y.shape[0]
    K = pi.shape[0]
    alpha_hat = np.zeros((T, K))
    c = np.zeros(T)

    # t = 0
    for j in range(K):
        alpha_hat[0, j] = pi[j] * np.exp(-0.5 * ((y[0] - mu[j]) / sigma[j])**2) / (
            sigma[j] * np.sqrt(2.0 * np.pi))
    c[0] = np.sum(alpha_hat[0])
    if c[0] > 0:
        alpha_hat[0] /= c[0]

    # t = 1, ..., T-1
    for t in range(1, T):
        for j in range(K):
            s = 0.0
            for i in range(K):
                s += alpha_hat[t - 1, i] * A[i, j]
            emission = np.exp(-0.5 * ((y[t] - mu[j]) / sigma[j])**2) / (
                sigma[j] * np.sqrt(2.0 * np.pi))
            alpha_hat[t, j] = s * emission
        c[t] = np.sum(alpha_hat[t])
        if c[t] > 0:
            alpha_hat[t] /= c[t]

    log_lik = np.sum(np.log(c + 1e-300))
    return alpha_hat, c, log_lik


@njit
def _backward_scaled(y: np.ndarray, c: np.ndarray, A: np.ndarray,
                     mu: np.ndarray, sigma: np.ndarray):
    """Scaled backward algorithm."""
    T = y.shape[0]
    K = A.shape[0]
    beta_hat = np.zeros((T, K))

    # t = T-1
    for j in range(K):
        beta_hat[T - 1, j] = 1.0

    # t = T-2, ..., 0
    for t in range(T - 2, -1, -1):
        for i in range(K):
            s = 0.0
            for j in range(K):
                emission = np.exp(-0.5 * ((y[t + 1] - mu[j]) / sigma[j])**2) / (
                    sigma[j] * np.sqrt(2.0 * np.pi))
                s += A[i, j] * emission * beta_hat[t + 1, j]
            beta_hat[t, i] = s
        if c[t + 1] > 0:
            beta_hat[t] /= c[t + 1]

    return beta_hat


@njit
def _viterbi(y: np.ndarray, pi: np.ndarray, A: np.ndarray,
             mu: np.ndarray, sigma: np.ndarray):
    """Viterbi decoding in log space. Returns most probable state sequence."""
    T = y.shape[0]
    K = pi.shape[0]
    log_delta = np.zeros((T, K))
    psi = np.zeros((T, K), dtype=np.int64)

    # Initialization
    for j in range(K):
        log_emission = -0.5 * ((y[0] - mu[j]) / sigma[j])**2 - np.log(
            sigma[j] * np.sqrt(2.0 * np.pi))
        log_delta[0, j] = np.log(pi[j] + 1e-300) + log_emission

    # Recursion
    for t in range(1, T):
        for j in range(K):
            log_emission = -0.5 * ((y[t] - mu[j]) / sigma[j])**2 - np.log(
                sigma[j] * np.sqrt(2.0 * np.pi))
            best_val = -np.inf
            best_i = 0
            for i in range(K):
                val = log_delta[t - 1, i] + np.log(A[i, j] + 1e-300)
                if val > best_val:
                    best_val = val
                    best_i = i
            log_delta[t, j] = best_val + log_emission
            psi[t, j] = best_i

    # Backtrack
    states = np.zeros(T, dtype=np.int64)
    states[T - 1] = np.argmax(log_delta[T - 1])
    for t in range(T - 2, -1, -1):
        states[t] = psi[t + 1, states[t + 1]]

    return states, log_delta


def baum_welch(y: np.ndarray, K: int, max_iter: int = 200,
               tol: float = 1e-6, n_restarts: int = 10,
               rng: np.random.Generator = None) -> tuple[HMMParams, float]:
    """
    Baum-Welch EM algorithm for Gaussian HMM.

    Parameters
    ----------
    y : (T,) observations
    K : number of hidden states
    max_iter : maximum EM iterations per restart
    tol : convergence tolerance on log-likelihood
    n_restarts : number of random initializations

    Returns
    -------
    best_params : HMMParams with highest log-likelihood
    best_ll : corresponding log-likelihood
    """
    if rng is None:
        rng = np.random.default_rng(42)

    T = y.shape[0]
    best_ll = -np.inf
    best_params = None

    for restart in range(n_restarts):
        # Random initialization
        pi = np.ones(K) / K
        A = rng.dirichlet(np.ones(K) * 5, size=K)

        # K-means-style initialization for emissions
        quantiles = np.quantile(y, np.linspace(0.1, 0.9, K))
        mu = quantiles + rng.normal(0, 0.01, K)
        sigma = np.full(K, np.std(y) / K) + np.abs(rng.normal(0, 0.01, K))

        prev_ll = -np.inf

        for iteration in range(max_iter):
            # --- E-step ---
            alpha_hat, c, log_lik = _forward_scaled(y, pi, A, mu, sigma)
            beta_hat = _backward_scaled(y, c, A, mu, sigma)

            # Posterior state probabilities: gamma_t(j)
            gamma = alpha_hat * beta_hat
            gamma /= gamma.sum(axis=1, keepdims=True) + 1e-300

            # Transition posteriors: xi_t(i, j)
            xi = np.zeros((T - 1, K, K))
            for t in range(T - 1):
                for i in range(K):
                    for j in range(K):
                        emission_j = norm.pdf(y[t + 1], mu[j], sigma[j])
                        xi[t, i, j] = (alpha_hat[t, i] * A[i, j] *
                                        emission_j * beta_hat[t + 1, j])
                denom = xi[t].sum()
                if denom > 0:
                    xi[t] /= denom

            # --- M-step ---
            pi = gamma[0]

            for i in range(K):
                gamma_sum_i = gamma[:-1, i].sum()
                if gamma_sum_i > 0:
                    for j in range(K):
                        A[i, j] = xi[:, i, j].sum() / gamma_sum_i

            for k in range(K):
                gamma_sum_k = gamma[:, k].sum()
                if gamma_sum_k > 1e-10:
                    mu[k] = (gamma[:, k] * y).sum() / gamma_sum_k
                    sigma[k] = np.sqrt(
                        (gamma[:, k] * (y - mu[k])**2).sum() / gamma_sum_k
                    )
                    sigma[k] = max(sigma[k], 1e-6)  # floor

            # Convergence check
            if abs(log_lik - prev_ll) < tol:
                break
            prev_ll = log_lik

        if log_lik > best_ll:
            best_ll = log_lik
            # Sort states by mean for identifiability
            order = np.argsort(mu)
            best_params = HMMParams(
                pi=pi[order],
                A=A[np.ix_(order, order)],
                mu=mu[order],
                sigma=sigma[order],
            )

    return best_params, best_ll


# ============================================================================
# 2. Bayesian Online Change Point Detection (BOCPD)
# ============================================================================

class BOCPDResult(NamedTuple):
    run_length_probs: np.ndarray  # (T, T+1) run length posterior
    changepoint_prob: np.ndarray  # (T,) probability of changepoint at each t
    mean_run_length: np.ndarray   # (T,) posterior mean run length


def bocpd_gaussian(y: np.ndarray, hazard_lambda: float = 200.0,
                   mu0: float = 0.0, kappa0: float = 1.0,
                   alpha0: float = 1.0, beta0: float = 1.0) -> BOCPDResult:
    """
    Bayesian Online Change Point Detection with Gaussian likelihood
    and Normal-Inverse-Gamma conjugate prior.

    Parameters
    ----------
    y : (T,) observations
    hazard_lambda : expected run length (geometric hazard 1/lambda)
    mu0, kappa0, alpha0, beta0 : NIG prior hyperparameters

    Returns
    -------
    BOCPDResult with run length posteriors and changepoint probabilities
    """
    T = len(y)
    H = 1.0 / hazard_lambda  # constant hazard

    # Run length probabilities: R[t, r] = P(r_t = r, y_{1:t})
    R = np.zeros((T + 1, T + 1))
    R[0, 0] = 1.0

    # Sufficient statistics for each run length
    # For NIG: track running mean, precision, alpha, beta
    mu_params = np.zeros(T + 1)
    kappa_params = np.zeros(T + 1)
    alpha_params = np.zeros(T + 1)
    beta_params = np.zeros(T + 1)

    mu_params[0] = mu0
    kappa_params[0] = kappa0
    alpha_params[0] = alpha0
    beta_params[0] = beta0

    changepoint_prob = np.zeros(T)
    mean_run_length = np.zeros(T)

    for t in range(T):
        x = y[t]

        # Predictive probability for each run length
        # Student-t: t_{2*alpha}(mu, beta*(kappa+1)/(alpha*kappa))
        pred_probs = np.zeros(t + 1)
        for r in range(t + 1):
            nu = 2.0 * alpha_params[r]
            pred_var = beta_params[r] * (kappa_params[r] + 1.0) / (
                alpha_params[r] * kappa_params[r])
            pred_scale = np.sqrt(pred_var)
            # Student-t pdf
            z = (x - mu_params[r]) / pred_scale
            from scipy.special import gammaln
            log_pred = (gammaln((nu + 1) / 2) - gammaln(nu / 2)
                        - 0.5 * np.log(nu * np.pi) - np.log(pred_scale)
                        - (nu + 1) / 2 * np.log(1 + z**2 / nu))
            pred_probs[r] = np.exp(log_pred)

        # Growth probabilities
        growth = R[t, :t + 1] * pred_probs * (1 - H)

        # Changepoint probability
        cp = np.sum(R[t, :t + 1] * pred_probs * H)

        # Update run length distribution
        R[t + 1, 0] = cp
        R[t + 1, 1:t + 2] = growth

        # Normalize
        evidence = R[t + 1, :t + 2].sum()
        if evidence > 0:
            R[t + 1, :t + 2] /= evidence

        changepoint_prob[t] = R[t + 1, 0]
        mean_run_length[t] = np.sum(
            np.arange(t + 2) * R[t + 1, :t + 2])

        # Update sufficient statistics (shift and add new run)
        new_mu = np.zeros(t + 2)
        new_kappa = np.zeros(t + 2)
        new_alpha = np.zeros(t + 2)
        new_beta = np.zeros(t + 2)

        # Run length 0: reset to prior
        new_mu[0] = mu0
        new_kappa[0] = kappa0
        new_alpha[0] = alpha0
        new_beta[0] = beta0

        # Run lengths 1, ..., t+1: update from previous
        for r in range(t + 1):
            old_kappa = kappa_params[r]
            old_mu = mu_params[r]
            old_alpha = alpha_params[r]
            old_beta = beta_params[r]

            new_kappa[r + 1] = old_kappa + 1
            new_mu[r + 1] = (old_kappa * old_mu + x) / (old_kappa + 1)
            new_alpha[r + 1] = old_alpha + 0.5
            new_beta[r + 1] = (old_beta +
                                old_kappa * (x - old_mu)**2 / (2 * (old_kappa + 1)))

        mu_params = new_mu
        kappa_params = new_kappa
        alpha_params = new_alpha
        beta_params = new_beta

    return BOCPDResult(
        run_length_probs=R,
        changepoint_prob=changepoint_prob,
        mean_run_length=mean_run_length,
    )


# ============================================================================
# 3. PELT (Pruned Exact Linear Time) Change-Point Detection
# ============================================================================

def pelt_gaussian(y: np.ndarray, penalty: float = None) -> list[int]:
    """
    PELT algorithm for Gaussian mean-shift change-point detection.

    Parameters
    ----------
    y : (T,) observations
    penalty : BIC penalty per change point (default: log(T))

    Returns
    -------
    changepoints : list of change-point indices
    """
    T = len(y)
    if penalty is None:
        penalty = np.log(T)

    # Precompute cumulative sums for O(1) segment cost
    cumsum = np.zeros(T + 1)
    cumsum_sq = np.zeros(T + 1)
    for t in range(T):
        cumsum[t + 1] = cumsum[t] + y[t]
        cumsum_sq[t + 1] = cumsum_sq[t] + y[t]**2

    def segment_cost(s: int, t: int) -> float:
        """Negative Gaussian log-likelihood for segment y[s:t]."""
        n = t - s
        if n <= 0:
            return 0.0
        mean_seg = (cumsum[t] - cumsum[s]) / n
        var_seg = (cumsum_sq[t] - cumsum_sq[s]) / n - mean_seg**2
        var_seg = max(var_seg, 1e-10)
        return n * np.log(var_seg)

    # DP
    F = np.full(T + 1, np.inf)
    F[0] = -penalty
    last_cp = np.zeros(T + 1, dtype=int)
    candidates = [0]

    for t_star in range(1, T + 1):
        # Find optimal last change point
        best_F = np.inf
        best_s = 0

        new_candidates = []
        for s in candidates:
            cost = F[s] + segment_cost(s, t_star) + penalty
            if cost < best_F:
                best_F = cost
                best_s = s
            # PELT pruning: keep s only if it could be useful later
            if F[s] + segment_cost(s, t_star) <= best_F:
                new_candidates.append(s)

        F[t_star] = best_F
        last_cp[t_star] = best_s
        new_candidates.append(t_star)
        candidates = new_candidates

    # Backtrack to find change points
    changepoints = []
    idx = T
    while idx > 0:
        cp = last_cp[idx]
        if cp > 0:
            changepoints.append(cp)
        idx = cp

    return sorted(changepoints)


# ============================================================================
# 4. Hamilton Filter for Markov-Switching Model
# ============================================================================

def hamilton_filter(y: np.ndarray, mu: np.ndarray, sigma: np.ndarray,
                    A: np.ndarray, pi: np.ndarray = None):
    """
    Hamilton (1989) filter for Markov-switching model.

    Parameters
    ----------
    y : (T,) observations
    mu : (K,) regime means
    sigma : (K,) regime standard deviations
    A : (K, K) transition matrix
    pi : (K,) initial distribution (default: stationary)

    Returns
    -------
    filtered_probs : (T, K) P(S_t = k | y_{1:t})
    predicted_probs : (T, K) P(S_t = k | y_{1:t-1})
    log_lik : scalar log-likelihood
    """
    T = len(y)
    K = len(mu)

    if pi is None:
        # Stationary distribution: solve pi = pi @ A
        eigenvalues, eigenvectors = np.linalg.eig(A.T)
        idx = np.argmin(np.abs(eigenvalues - 1.0))
        pi = np.real(eigenvectors[:, idx])
        pi = pi / pi.sum()

    filtered_probs = np.zeros((T, K))
    predicted_probs = np.zeros((T, K))
    log_lik = 0.0

    xi_prev = pi.copy()

    for t in range(T):
        # Prediction step: P(S_t | y_{1:t-1})
        if t == 0:
            xi_pred = pi.copy()
        else:
            xi_pred = A.T @ xi_prev

        predicted_probs[t] = xi_pred

        # Likelihood for each state
        f_y = norm.pdf(y[t], mu, sigma)

        # Joint: P(S_t = k, y_t | y_{1:t-1})
        joint = xi_pred * f_y

        # Marginal: P(y_t | y_{1:t-1})
        marginal = joint.sum()
        log_lik += np.log(marginal + 1e-300)

        # Update step: P(S_t | y_{1:t})
        xi_prev = joint / (marginal + 1e-300)
        filtered_probs[t] = xi_prev

    return filtered_probs, predicted_probs, log_lik


# ============================================================================
# 5. Regime-Conditional Allocation & Adaptive Kelly
# ============================================================================

def regime_conditional_allocation(
    regime_probs: np.ndarray,
    regime_portfolios: np.ndarray,
) -> np.ndarray:
    """
    Blend regime-specific portfolios using filtered regime probabilities.

    Parameters
    ----------
    regime_probs : (T, K) filtered regime probabilities
    regime_portfolios : (K, N) optimal portfolio weights per regime

    Returns
    -------
    weights : (T, N) time-varying portfolio weights
    """
    return regime_probs @ regime_portfolios


def adaptive_kelly(mu: np.ndarray, sigma: np.ndarray,
                   regime_probs: np.ndarray) -> np.ndarray:
    """
    Adaptive Kelly criterion with regime-uncertainty penalty.

    Parameters
    ----------
    mu : (K,) regime means
    sigma : (K,) regime standard deviations
    regime_probs : (T, K) filtered regime probabilities

    Returns
    -------
    kelly_fractions : (T,) time-varying Kelly fractions
    """
    K = mu.shape[0]

    # Kelly fraction per regime
    f_k = mu / (sigma**2)

    # Blended Kelly
    f_blend = regime_probs @ f_k

    # Entropy-based uncertainty penalty
    eps = 1e-10
    entropy = -np.sum(regime_probs * np.log(regime_probs + eps), axis=1)
    max_entropy = np.log(K)
    normalized_entropy = entropy / max_entropy

    # Scale down when uncertain
    f_adjusted = f_blend * (1.0 - normalized_entropy)

    return f_adjusted


# ============================================================================
# 6. Example: Full Regime Detection Pipeline
# ============================================================================

def demo_regime_detection():
    """End-to-end regime detection on simulated data."""
    rng = np.random.default_rng(42)

    # Simulate 2-regime returns
    T = 1000
    true_states = np.zeros(T, dtype=int)
    state = 0
    for t in range(1, T):
        if state == 0:
            state = 1 if rng.random() < 0.02 else 0
        else:
            state = 0 if rng.random() < 0.05 else 1
        true_states[t] = state

    mu_true = np.array([0.0005, -0.001])    # daily: +12.5% ann, -25% ann
    sigma_true = np.array([0.01, 0.025])     # 16% ann, 40% ann

    y = np.array([rng.normal(mu_true[s], sigma_true[s]) for s in true_states])

    # 1. Fit HMM via Baum-Welch
    params, ll = baum_welch(y, K=2, n_restarts=5, rng=rng)
    print(f"HMM log-likelihood: {ll:.2f}")
    print(f"  Means: {params.mu}")
    print(f"  Sigmas: {params.sigma}")
    print(f"  Transition matrix:\n{params.A}")

    # 2. Viterbi decoding
    states_viterbi, _ = _viterbi(y, params.pi, params.A, params.mu, params.sigma)
    accuracy = np.mean(states_viterbi == true_states)
    print(f"  Viterbi accuracy: {accuracy:.3f}")

    # 3. Hamilton filter
    filtered, _, _ = hamilton_filter(y, params.mu, params.sigma, params.A)
    print(f"  Hamilton filter shape: {filtered.shape}")

    # 4. BOCPD
    result = bocpd_gaussian(y, hazard_lambda=100.0)
    cp_times = np.where(result.changepoint_prob > 0.3)[0]
    print(f"  BOCPD detected {len(cp_times)} potential change points")

    # 5. PELT
    changepoints = pelt_gaussian(y, penalty=2.0 * np.log(T))
    print(f"  PELT detected {len(changepoints)} change points")

    # 6. Adaptive Kelly
    kelly = adaptive_kelly(params.mu, params.sigma, filtered)
    print(f"  Adaptive Kelly range: [{kelly.min():.4f}, {kelly.max():.4f}]")


if __name__ == "__main__":
    demo_regime_detection()
```

---

## 15. Implementation: C++

```cpp
/*
 * Module 33: HMM Forward-Backward Algorithm — C++ Implementation
 * Requires: Eigen 3.4+
 * Compile:  g++ -std=c++20 -O3 -march=native -I/path/to/eigen -o hmm hmm.cpp
 */

#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <random>
#include <cassert>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::ArrayXd;

static constexpr double LOG_2PI = 1.8378770664093453;

// ---------------------------------------------------------------------------
// Gaussian PDF (log and standard)
// ---------------------------------------------------------------------------

inline double gaussian_log_pdf(double x, double mu, double sigma) {
    double z = (x - mu) / sigma;
    return -0.5 * (LOG_2PI + 2.0 * std::log(sigma) + z * z);
}

inline double gaussian_pdf(double x, double mu, double sigma) {
    return std::exp(gaussian_log_pdf(x, mu, sigma));
}

// ---------------------------------------------------------------------------
// HMM Parameters
// ---------------------------------------------------------------------------

struct HMMParams {
    VectorXd pi;       // (K,) initial distribution
    MatrixXd A;        // (K, K) transition matrix
    VectorXd mu;       // (K,) emission means
    VectorXd sigma;    // (K,) emission std devs
    int K;             // number of states

    HMMParams(int k) : K(k), pi(k), A(k, k), mu(k), sigma(k) {
        pi.setConstant(1.0 / k);
        A.setConstant(1.0 / k);
        mu.setZero();
        sigma.setOnes();
    }
};

// ---------------------------------------------------------------------------
// Forward-Backward Result
// ---------------------------------------------------------------------------

struct ForwardBackwardResult {
    MatrixXd alpha;    // (T, K) scaled forward variables
    MatrixXd beta;     // (T, K) scaled backward variables
    MatrixXd gamma;    // (T, K) smoothed posteriors P(S_t=k | y_{1:T})
    VectorXd scale;    // (T,) scaling factors
    double log_lik;    // log P(y | lambda)
};

// ---------------------------------------------------------------------------
// Scaled Forward Algorithm
// ---------------------------------------------------------------------------

ForwardBackwardResult forward_backward(
    const std::vector<double>& y,
    const HMMParams& params
) {
    const int T = static_cast<int>(y.size());
    const int K = params.K;

    MatrixXd alpha(T, K);
    MatrixXd beta(T, K);
    MatrixXd gamma(T, K);
    VectorXd c(T);  // scaling factors

    // ======= Forward pass =======

    // t = 0: alpha_0(j) = pi_j * f(y_0 | theta_j)
    for (int j = 0; j < K; ++j) {
        alpha(0, j) = params.pi(j) *
            gaussian_pdf(y[0], params.mu(j), params.sigma(j));
    }
    c(0) = alpha.row(0).sum();
    if (c(0) > 0) alpha.row(0) /= c(0);

    // t = 1, ..., T-1
    for (int t = 1; t < T; ++t) {
        for (int j = 0; j < K; ++j) {
            double sum = 0.0;
            for (int i = 0; i < K; ++i) {
                sum += alpha(t - 1, i) * params.A(i, j);
            }
            alpha(t, j) = sum *
                gaussian_pdf(y[t], params.mu(j), params.sigma(j));
        }
        c(t) = alpha.row(t).sum();
        if (c(t) > 0) alpha.row(t) /= c(t);
    }

    // Log-likelihood
    double log_lik = 0.0;
    for (int t = 0; t < T; ++t) {
        log_lik += std::log(c(t) + 1e-300);
    }

    // ======= Backward pass =======

    // t = T-1
    beta.row(T - 1).setOnes();

    // t = T-2, ..., 0
    for (int t = T - 2; t >= 0; --t) {
        for (int i = 0; i < K; ++i) {
            double sum = 0.0;
            for (int j = 0; j < K; ++j) {
                sum += params.A(i, j) *
                    gaussian_pdf(y[t + 1], params.mu(j), params.sigma(j)) *
                    beta(t + 1, j);
            }
            beta(t, i) = sum;
        }
        if (c(t + 1) > 0) beta.row(t) /= c(t + 1);
    }

    // ======= Smoothed posteriors =======

    for (int t = 0; t < T; ++t) {
        gamma.row(t) = alpha.row(t).array() * beta.row(t).array();
        double row_sum = gamma.row(t).sum();
        if (row_sum > 0) gamma.row(t) /= row_sum;
    }

    return {alpha, beta, gamma, c, log_lik};
}

// ---------------------------------------------------------------------------
// Viterbi Decoding
// ---------------------------------------------------------------------------

std::vector<int> viterbi(
    const std::vector<double>& y,
    const HMMParams& params
) {
    const int T = static_cast<int>(y.size());
    const int K = params.K;

    MatrixXd log_delta(T, K);
    Eigen::MatrixXi psi(T, K);

    // Initialization
    for (int j = 0; j < K; ++j) {
        log_delta(0, j) = std::log(params.pi(j) + 1e-300) +
            gaussian_log_pdf(y[0], params.mu(j), params.sigma(j));
    }

    // Recursion
    for (int t = 1; t < T; ++t) {
        for (int j = 0; j < K; ++j) {
            double best_val = -std::numeric_limits<double>::infinity();
            int best_i = 0;
            for (int i = 0; i < K; ++i) {
                double val = log_delta(t - 1, i) +
                    std::log(params.A(i, j) + 1e-300);
                if (val > best_val) {
                    best_val = val;
                    best_i = i;
                }
            }
            log_delta(t, j) = best_val +
                gaussian_log_pdf(y[t], params.mu(j), params.sigma(j));
            psi(t, j) = best_i;
        }
    }

    // Backtrack
    std::vector<int> states(T);
    int best_final = 0;
    double best_final_val = log_delta(T - 1, 0);
    for (int j = 1; j < K; ++j) {
        if (log_delta(T - 1, j) > best_final_val) {
            best_final_val = log_delta(T - 1, j);
            best_final = j;
        }
    }
    states[T - 1] = best_final;
    for (int t = T - 2; t >= 0; --t) {
        states[t] = psi(t + 1, states[t + 1]);
    }

    return states;
}

// ---------------------------------------------------------------------------
// Demonstration
// ---------------------------------------------------------------------------

int main() {
    // Set up a 2-state HMM
    HMMParams params(2);
    params.pi << 0.8, 0.2;
    params.A << 0.97, 0.03,
                0.05, 0.95;
    params.mu << 0.0005, -0.001;
    params.sigma << 0.01, 0.025;

    // Simulate data
    std::mt19937_64 rng(42);
    std::uniform_real_distribution<double> unif(0.0, 1.0);

    const int T = 1000;
    std::vector<double> y(T);
    std::vector<int> true_states(T);
    int state = 0;

    for (int t = 0; t < T; ++t) {
        true_states[t] = state;
        std::normal_distribution<double> dist(
            params.mu(state), params.sigma(state));
        y[t] = dist(rng);

        // Transition
        double u = unif(rng);
        state = (u < params.A(state, 0)) ? 0 : 1;
    }

    // Run forward-backward
    auto result = forward_backward(y, params);

    std::cout << "=== Forward-Backward ===" << std::endl;
    std::cout << "  Log-likelihood: " << result.log_lik << std::endl;
    std::cout << "  Gamma[0]: " << result.gamma.row(0) << std::endl;
    std::cout << "  Gamma[T/2]: " << result.gamma.row(T / 2) << std::endl;

    // Run Viterbi
    auto decoded = viterbi(y, params);
    int correct = 0;
    for (int t = 0; t < T; ++t) {
        if (decoded[t] == true_states[t]) ++correct;
    }
    double accuracy = static_cast<double>(correct) / T;

    std::cout << "\n=== Viterbi Decoding ===" << std::endl;
    std::cout << "  Accuracy: " << accuracy * 100.0 << "%" << std::endl;

    // Count regime transitions
    int transitions = 0;
    for (int t = 1; t < T; ++t) {
        if (decoded[t] != decoded[t - 1]) ++transitions;
    }
    std::cout << "  Detected transitions: " << transitions << std::endl;

    // Average regime duration
    int bull_count = 0, bear_count = 0;
    for (int t = 0; t < T; ++t) {
        if (decoded[t] == 0) ++bull_count;
        else ++bear_count;
    }
    std::cout << "  Bull days: " << bull_count
              << ", Bear days: " << bear_count << std::endl;

    return 0;
}
```

---

## 16. Exercises

### Foundational

**Exercise 33.1.** For a 2-state HMM with transition matrix $\mathbf{A} = \begin{pmatrix} 0.97 & 0.03 \\ 0.05 & 0.95 \end{pmatrix}$, compute:
- (a) The stationary distribution $\boldsymbol{\pi}^*$.
- (b) Expected duration in each state.
- (c) The probability of being in state 1 at time $t = 10$ given $S_0 = 2$.

**Exercise 33.2.** Derive the backward variable recursion $\beta_t(i) = \sum_j a_{ij} f(y_{t+1} \mid \boldsymbol{\theta}_j) \beta_{t+1}(j)$ from the definition $\beta_t(i) = P(y_{t+1:T} \mid S_t = i, \lambda)$.

**Exercise 33.3.** Show that the Viterbi algorithm's backtracking produces a valid state sequence (i.e., $a_{s_t^*, s_{t+1}^*} > 0$ for all $t$) whenever the transition matrix has no zero entries.

**Exercise 33.4.** Prove that the Baum-Welch update for $a_{ij}$ maximizes the expected complete-data log-likelihood $Q(\lambda, \lambda^{(n)})$ subject to $\sum_j a_{ij} = 1$.

### Computational

**Exercise 33.5.** Implement the BOCPD algorithm and apply it to the following signal: 500 samples from $\mathcal{N}(0, 1)$, followed by 300 from $\mathcal{N}(2, 1)$, followed by 200 from $\mathcal{N}(-1, 1.5)$. Plot the run-length posterior as a heatmap and the changepoint probability time series. Experiment with different hazard rates $\lambda \in \{50, 100, 200, 500\}$.

**Exercise 33.6.** Download 20 years of S&P 500 daily returns. Fit Gaussian HMMs with $K \in \{2, 3, 4\}$ states. For each:
- Report AIC and BIC
- Plot filtered regime probabilities overlaid on the price series
- Shade the 2008 financial crisis, 2020 COVID crash, and 2022 drawdown
- Which $K$ best captures these events?

**Exercise 33.7.** Implement the PELT algorithm. Apply it to the squared returns of the S&P 500 (a proxy for realized variance). Compare the detected change points against known events (LTCM, dot-com, GFC, COVID). Vary the penalty $\beta$ and produce a CROPS-style elbow plot of number of change points vs. penalty.

**Exercise 33.8.** Build a regime-conditional allocation strategy:
- Fit a 2-state HMM to S&P 500 returns
- Bull regime: 80% equities, 20% bonds. Bear regime: 20% equities, 80% bonds
- Blend using filtered probabilities
- Backtest against buy-and-hold, reporting annualized return, volatility, Sharpe ratio, and maximum drawdown
- Add a 5-day implementation lag and re-evaluate

### Advanced

**Exercise 33.9.** Implement the adaptive Kelly criterion with entropy-based regime uncertainty penalty. Simulate 10,000 paths from a 2-state HMM with known parameters. Compare the terminal wealth distribution of:
- (a) Full Kelly assuming the true regime is known
- (b) Adaptive Kelly using filtered probabilities
- (c) Adaptive Kelly with entropy penalty
- (d) Half-Kelly (constant $f/2$)
Compute the probability of ruin (drawdown > 50%) for each.

**Exercise 33.10.** Build an ensemble regime detector combining: (i) a 2-state HMM, (ii) BOCPD, (iii) a 200-day moving average crossover signal, (iv) VIX level above/below 20. Implement both hard voting and soft probability aggregation (inverse-Brier-score weighted). Evaluate on 2000--2024 data using a forward-looking Sharpe ratio as the ground truth regime label.

---

## 17. Summary and Concept Map

```mermaid
graph TD
    OBS[Observed Returns] --> HMM[Hidden Markov Models]
    OBS --> CPD[Change-Point Detection]
    OBS --> SBT[Structural Break Tests]

    HMM --> FB[Forward-Backward<br/>Algorithm]
    HMM --> VIT[Viterbi Decoding]
    HMM --> BW[Baum-Welch EM]
    HMM --> MSEL[Model Selection<br/>AIC / BIC]

    FB --> FILT[Filtered Probs<br/>P(S_t | y_{1:t})]
    FB --> SMOOTH[Smoothed Probs<br/>P(S_t | y_{1:T})]

    CPD --> PELT[PELT<br/>Offline Exact]
    CPD --> BOCPD[BOCPD<br/>Online Bayesian]
    CPD --> CUSUM[CUSUM Charts]

    SBT --> BAIP[Bai-Perron<br/>sup-F Test]
    SBT --> QA[Quandt-Andrews]

    HMM --> MSM[Markov-Switching<br/>Models]
    MSM --> HAM[Hamilton Filter]
    MSM --> MSG[MS-GARCH]

    OBS --> SPEC[Spectral Methods]
    SPEC --> WAV[Wavelet Variance]
    SPEC --> HHT[Hilbert-Huang<br/>Transform]

    FILT --> ENS[Ensemble<br/>Detection]
    BOCPD --> ENS
    WAV --> ENS

    ENS --> ADAPT[Adaptive Strategies]
    FILT --> ADAPT
    ADAPT --> RCA[Regime-Conditional<br/>Allocation]
    ADAPT --> AKELLY[Adaptive Kelly<br/>Criterion]
    ADAPT --> EWMA[Adaptive Forgetting<br/>Factors]
    ADAPT --> META[Meta-Strategy<br/>Selection]

    HMM --> TRANS[Transition<br/>Dynamics]
    TRANS --> DUR[Duration Models]
    TRANS --> HSMM[Semi-Markov<br/>Extensions]

    style HMM fill:#4a90d9,color:#fff
    style ADAPT fill:#27ae60,color:#fff
    style BOCPD fill:#e67e22,color:#fff
    style ENS fill:#8e44ad,color:#fff
```

This module covered the complete toolkit for detecting and exploiting market regimes:

1. **Hidden Markov Models** provide the foundational probabilistic framework. The forward-backward algorithm computes exact posterior state probabilities in $O(TK^2)$; Viterbi finds the most probable path; Baum-Welch learns parameters via EM.

2. **Change-point detection** complements HMMs by identifying structural breaks. PELT solves the offline problem exactly in $O(T)$ expected time. BOCPD provides elegant online detection with a Bayesian run-length posterior.

3. **Markov-switching models** (Hamilton) embed regime dynamics into econometric models, allowing regime-dependent means, variances, and autoregressive dynamics simultaneously.

4. **Spectral methods** (wavelets, HHT) offer non-parametric alternatives that do not assume a specific number of regimes.

5. **Ensemble methods** combine heterogeneous detectors for robustness, addressing the fundamental challenge that no single method dominates.

6. **Adaptive strategies** translate regime information into action: regime-conditional allocation, adaptive Kelly sizing with entropy-based uncertainty penalization, and meta-strategy selection.

The central tension in regime detection for trading is the **speed-accuracy trade-off**: smoothed posteriors (using future data) are accurate but unusable in real time; filtered probabilities are causal but noisy. Practical systems use filtered probabilities with conservative position sizing to manage the inherent regime classification uncertainty.

---

*Next: [Module 34 — Alternative Data & Feature Engineering](../advanced-alpha/34_alt_data_features.md)*

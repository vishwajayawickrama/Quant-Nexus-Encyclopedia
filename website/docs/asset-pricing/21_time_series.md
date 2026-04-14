# Module 21: Time Series Analysis for Quantitative Finance

> **Prerequisites:** Modules 01 (Mathematical Foundations), 02 (Probability & Stochastic Processes), 03 (Statistics & Estimation), 09 (Linear Algebra for Finance)
> **Builds toward:** Modules 22 (Kalman Filters), 25 (Factor Models), 26 (Risk Models), 33 (Algorithmic Trading Systems)

---

## Table of Contents

1. [Stationarity and Unit Roots](#1-stationarity-and-unit-roots)
2. [ARMA Models](#2-arma-models)
3. [ARIMA and Seasonal Models](#3-arima-and-seasonal-models)
4. [Volatility Modeling](#4-volatility-modeling)
5. [Realized Volatility and High-Frequency Measures](#5-realized-volatility-and-high-frequency-measures)
6. [HAR-RV Model](#6-har-rv-model)
7. [Cointegration and Error Correction](#7-cointegration-and-error-correction)
8. [Structural Breaks](#8-structural-breaks)
9. [Long Memory Processes](#9-long-memory-processes)
10. [Implementation: Python](#10-implementation-python)
11. [Implementation: C++](#11-implementation-cpp)
12. [Exercises](#12-exercises)
13. [Summary and Concept Map](#13-summary-and-concept-map)

---

## 1. Stationarity and Unit Roots

Stationarity is the foundational assumption underlying virtually all classical time series models. A process that wanders without bound cannot be meaningfully characterized by fixed distributional parameters, and forecasting becomes ill-defined. In financial markets, the question of whether an asset price is stationary or contains a unit root determines the entire modeling strategy: returns versus levels, cointegration versus spurious regression, mean-reversion versus trend-following.

### 1.1 Strict (Strong) Stationarity

A stochastic process $\{X_t\}$ is **strictly stationary** if the joint distribution of any collection of random variables is invariant under time shifts. Formally, for all $k$, all $t_1, t_2, \ldots, t_k$, and all $\tau$:

$$
F_{X_{t_1}, X_{t_2}, \ldots, X_{t_k}}(x_1, x_2, \ldots, x_k) = F_{X_{t_1+\tau}, X_{t_2+\tau}, \ldots, X_{t_k+\tau}}(x_1, x_2, \ldots, x_k)
$$

This requires that the *entire* probability structure is time-invariant. In practice, strict stationarity is too strong a requirement to test empirically and is rarely assumed directly.

### 1.2 Weak (Wide-Sense / Covariance) Stationarity

A process $\{X_t\}$ is **weakly stationary** (also called covariance stationary or second-order stationary) if:

1. **Constant mean:** $\mathbb{E}[X_t] = \mu$ for all $t$
2. **Constant variance:** $\text{Var}(X_t) = \sigma^2 < \infty$ for all $t$
3. **Autocovariance depends only on lag:** $\text{Cov}(X_t, X_{t+h}) = \gamma(h)$ for all $t$

The autocovariance function $\gamma(h)$ is the fundamental characterization of a weakly stationary process. It satisfies $\gamma(0) = \sigma^2$, $\gamma(-h) = \gamma(h)$ (symmetry), and $|\gamma(h)| \leq \gamma(0)$ (Cauchy-Schwarz). The autocorrelation function (ACF) is defined as:

$$
\rho(h) = \frac{\gamma(h)}{\gamma(0)}
$$

For Gaussian processes, weak stationarity implies strict stationarity because the Gaussian distribution is fully characterized by its first two moments.

### 1.3 Augmented Dickey-Fuller (ADF) Test

The most widely used unit root test in finance. Consider the AR(1) model:

$$
X_t = \phi X_{t-1} + \varepsilon_t, \quad \varepsilon_t \sim \text{WN}(0, \sigma^2)
$$

The process has a unit root when $\phi = 1$. Rewrite by subtracting $X_{t-1}$ from both sides:

$$
\Delta X_t = (\phi - 1) X_{t-1} + \varepsilon_t = \delta X_{t-1} + \varepsilon_t
$$

where $\delta = \phi - 1$. Testing $H_0: \phi = 1$ is equivalent to testing $H_0: \delta = 0$.

**Derivation of the test statistic.** Under $H_0: \delta = 0$, we have $\Delta X_t = \varepsilon_t$. The OLS estimator of $\delta$ is:

$$
\hat{\delta} = \frac{\sum_{t=2}^{T} X_{t-1} \Delta X_t}{\sum_{t=2}^{T} X_{t-1}^2}
$$

Under $H_0$, $X_t$ is a random walk, so $X_{t-1} = \sum_{i=1}^{t-1} \varepsilon_i$. The numerator becomes:

$$
\sum_{t=2}^{T} X_{t-1} \varepsilon_t
$$

Using the identity $X_{t-1} \varepsilon_t = \frac{1}{2}(X_t^2 - X_{t-1}^2 - \varepsilon_t^2)$, the numerator telescopes to:

$$
\sum_{t=2}^{T} X_{t-1} \varepsilon_t = \frac{1}{2}\left(X_T^2 - X_1^2 - \sum_{t=2}^{T} \varepsilon_t^2\right)
$$

The ADF test statistic (the Dickey-Fuller $\tau$-statistic) is:

$$
\text{ADF} = \frac{\hat{\delta}}{\text{SE}(\hat{\delta})}
$$

Critically, under $H_0$, $\hat{\delta}$ does **not** follow a normal distribution. The denominator involves $\sum X_{t-1}^2$, which when properly scaled converges to a functional of Brownian motion:

$$
T \hat{\delta} \xrightarrow{d} \frac{\frac{1}{2}[W(1)^2 - 1]}{\int_0^1 W(r)^2 \, dr}
$$

where $W(\cdot)$ is a standard Wiener process. This is the **Dickey-Fuller distribution**, which is left-skewed and has critical values tabulated by simulation (e.g., 5% critical value $\approx -2.86$ for the model with intercept).

The *augmented* version includes $p$ lagged differences to account for serial correlation:

$$
\Delta X_t = \alpha + \delta X_{t-1} + \sum_{j=1}^{p} \beta_j \Delta X_{t-j} + \varepsilon_t
$$

Lag length $p$ is selected via information criteria (AIC, BIC) or sequential testing.

### 1.4 KPSS Test

The Kwiatkowski-Phillips-Schmidt-Shin test reverses the hypotheses: $H_0$ is stationarity (around a deterministic trend). The model decomposes $X_t$ into a deterministic trend, random walk, and stationary error:

$$
X_t = \xi t + r_t + \varepsilon_t, \quad r_t = r_{t-1} + u_t, \quad u_t \sim \text{WN}(0, \sigma_u^2)
$$

Under $H_0: \sigma_u^2 = 0$, so $r_t$ is constant and $X_t$ is trend-stationary. The test statistic is:

$$
\text{KPSS} = \frac{T^{-2} \sum_{t=1}^{T} S_t^2}{\hat{\sigma}_\varepsilon^2}
$$

where $S_t = \sum_{i=1}^{t} \hat{\varepsilon}_i$ is the partial sum of OLS residuals and $\hat{\sigma}_\varepsilon^2$ is a long-run variance estimator (using a kernel such as Bartlett). Using both ADF and KPSS together provides a more robust assessment: if ADF rejects but KPSS does not, conclude stationarity; if ADF does not reject but KPSS does, conclude unit root; conflicting results suggest fractional integration or structural breaks.

### 1.5 Phillips-Perron Test

The Phillips-Perron test modifies the Dickey-Fuller $t$-statistic nonparametrically to handle serial correlation and heteroskedasticity, avoiding the need to specify a lag length. It uses the Newey-West long-run variance estimator to correct the standard DF statistic:

$$
Z_t = t_{\hat{\delta}} \cdot \frac{\hat{\sigma}_\varepsilon}{\hat{\sigma}_{\text{lr}}} - \frac{T(\hat{\sigma}_{\text{lr}}^2 - \hat{\sigma}_\varepsilon^2) \cdot \text{SE}(\hat{\delta})}{2 \hat{\sigma}_{\text{lr}} \sqrt{\sum X_{t-1}^2}}
$$

The asymptotic distribution is the same Dickey-Fuller distribution, so the same critical values apply. The PP test is more robust than ADF to general forms of heteroskedasticity in $\varepsilon_t$ but can have severe size distortions in finite samples with MA roots near $-1$.

---

## 2. ARMA Models

Autoregressive Moving Average models are the workhorses of linear time series analysis. They describe a stationary process as a linear combination of its own past values and past innovation shocks.

### 2.1 AR(1) Process: Autocovariance Derivation

The AR(1) model is $X_t = \phi X_{t-1} + \varepsilon_t$ where $|\phi| < 1$ and $\varepsilon_t \sim \text{WN}(0, \sigma^2)$.

**Variance.** Multiply both sides by $X_t$ and take expectations:

$$
\gamma(0) = \mathbb{E}[X_t^2] = \phi^2 \mathbb{E}[X_{t-1}^2] + 2\phi \underbrace{\mathbb{E}[X_{t-1}\varepsilon_t]}_{=0} + \sigma^2
$$

Since $\varepsilon_t$ is independent of $X_{t-1}$ (causal representation), $\mathbb{E}[X_{t-1}\varepsilon_t] = 0$, and by stationarity $\mathbb{E}[X_t^2] = \mathbb{E}[X_{t-1}^2] = \gamma(0)$:

$$
\gamma(0) = \phi^2 \gamma(0) + \sigma^2 \implies \gamma(0) = \frac{\sigma^2}{1 - \phi^2}
$$

**Autocovariance at lag $h$.** Multiply $X_t = \phi X_{t-1} + \varepsilon_t$ by $X_{t-h}$ and take expectations for $h \geq 1$:

$$
\gamma(h) = \phi \gamma(h-1) + \underbrace{\mathbb{E}[\varepsilon_t X_{t-h}]}_{=0} = \phi \gamma(h-1)
$$

By recursion:

$$
\gamma(h) = \phi^h \gamma(0) = \frac{\phi^h \sigma^2}{1 - \phi^2}
$$

The ACF is therefore $\rho(h) = \phi^h$, which decays geometrically. The partial autocorrelation function (PACF) has $\alpha(1) = \phi$ and $\alpha(h) = 0$ for $h > 1$ --- this sharp cutoff after lag 1 is the identifying signature of an AR(1).

### 2.2 MA(1) Process: Autocovariance Derivation

The MA(1) model is $X_t = \varepsilon_t + \theta \varepsilon_{t-1}$ where $\varepsilon_t \sim \text{WN}(0, \sigma^2)$.

**Variance:**

$$
\gamma(0) = \mathbb{E}[(\varepsilon_t + \theta \varepsilon_{t-1})^2] = \sigma^2 + \theta^2 \sigma^2 = \sigma^2(1 + \theta^2)
$$

**Lag-1 autocovariance:**

$$
\gamma(1) = \mathbb{E}[(\varepsilon_t + \theta \varepsilon_{t-1})(\varepsilon_{t-1} + \theta \varepsilon_{t-2})] = \theta \sigma^2
$$

**Lag $h \geq 2$:** $\gamma(h) = 0$, since $\varepsilon_t$ and $\varepsilon_{t-h}$ share no common terms.

Thus the ACF of MA(1) has a sharp cutoff after lag 1:

$$
\rho(1) = \frac{\theta}{1 + \theta^2}, \quad \rho(h) = 0 \text{ for } h \geq 2
$$

The PACF of an MA(1) decays geometrically (tails off), which is the dual of the AR(1) pattern.

### 2.3 General ARMA(p,q) and Wold Decomposition

The general ARMA(p,q) model is:

$$
X_t - \phi_1 X_{t-1} - \cdots - \phi_p X_{t-p} = \varepsilon_t + \theta_1 \varepsilon_{t-1} + \cdots + \theta_q \varepsilon_{t-q}
$$

In lag operator notation: $\Phi(L) X_t = \Theta(L) \varepsilon_t$, where $\Phi(L) = 1 - \phi_1 L - \cdots - \phi_p L^p$ and $\Theta(L) = 1 + \theta_1 L + \cdots + \theta_q L^q$.

**Wold Decomposition Theorem.** Every covariance-stationary process $\{X_t\}$ with zero mean can be written as:

$$
X_t = \sum_{j=0}^{\infty} \psi_j \varepsilon_{t-j} + V_t
$$

where $\psi_0 = 1$, $\sum_{j=0}^{\infty} \psi_j^2 < \infty$, $\varepsilon_t$ is white noise, and $V_t$ is a deterministic component (linearly deterministic process). The $\psi_j$ are the MA($\infty$) coefficients obtained from $\Psi(L) = \Phi(L)^{-1} \Theta(L)$. The Wold decomposition provides the theoretical justification for ARMA modeling: any stationary process can be approximated arbitrarily well by an ARMA model of sufficiently high order.

### 2.4 Box-Jenkins Identification

The Box-Jenkins methodology is a systematic three-step procedure: *identification*, *estimation*, *diagnostic checking*.

**ACF/PACF patterns for identification:**

| Model   | ACF                          | PACF                         |
|---------|------------------------------|------------------------------|
| AR(p)   | Tails off (geometric/damped) | Cuts off after lag $p$       |
| MA(q)   | Cuts off after lag $q$       | Tails off (geometric/damped) |
| ARMA(p,q) | Tails off                  | Tails off                    |

For ARMA models, identification is aided by the extended autocorrelation function (EACF) or information criteria (AIC, BIC) over a grid of $(p,q)$ values. The BIC is consistent (selects the true order asymptotically) while the AIC tends to overfit but provides better out-of-sample forecasting in finite samples.

---

## 3. ARIMA and Seasonal Models

### 3.1 Integrated Processes

A process $\{X_t\}$ is **integrated of order $d$**, written $X_t \sim I(d)$, if it requires $d$ differences to become stationary. The ARIMA(p,d,q) model specifies:

$$
\Phi(L)(1 - L)^d X_t = \Theta(L) \varepsilon_t
$$

For financial log-prices, the random walk model $P_t = P_t-1 + \varepsilon_t$ is ARIMA(0,1,0), and log-returns $r_t = \Delta \log P_t$ are (approximately) stationary.

### 3.2 SARIMA

Seasonal patterns appear in many economic time series (quarterly GDP, monthly CPI). The SARIMA$(p,d,q) \times (P,D,Q)_s$ model is:

$$
\Phi(L)\, \Phi_s(L^s)\, (1-L)^d\, (1-L^s)^D\, X_t = \Theta(L)\, \Theta_s(L^s)\, \varepsilon_t
$$

where $\Phi_s(L^s) = 1 - \Phi_1 L^s - \cdots - \Phi_P L^{Ps}$ is the seasonal AR polynomial and $\Theta_s(L^s) = 1 + \Theta_1 L^s + \cdots + \Theta_Q L^{Qs}$ is the seasonal MA polynomial, with $s$ being the seasonal period (e.g., $s=12$ for monthly data with annual seasonality). Identification uses the seasonal ACF/PACF at lags $s, 2s, 3s, \ldots$.

---

## 4. Volatility Modeling

Financial returns exhibit several stylized facts that linear models cannot capture: volatility clustering ("large changes tend to be followed by large changes"), fat tails (excess kurtosis), and the leverage effect (negative returns increase future volatility more than positive returns of the same magnitude). This section develops the GARCH family of models that address these features.

### 4.1 ARCH (Engle, 1982)

The Autoregressive Conditional Heteroskedasticity model was introduced by Robert Engle. The return $r_t$ has a conditional mean $\mu_t$ and the innovation $a_t = r_t - \mu_t$ satisfies:

$$
a_t = \sigma_t z_t, \quad z_t \sim \text{i.i.d.}(0, 1)
$$

The ARCH(q) model specifies the conditional variance as:

$$
\sigma_t^2 = \alpha_0 + \alpha_1 a_{t-1}^2 + \alpha_2 a_{t-2}^2 + \cdots + \alpha_q a_{t-q}^2
$$

**Derivation of conditional variance.** The key insight is that while $\mathbb{E}[a_t] = 0$ and $\text{Var}(a_t) = \text{const}$ (unconditionally), the *conditional* variance varies over time:

$$
\text{Var}(a_t | \mathcal{F}_{t-1}) = \mathbb{E}[a_t^2 | \mathcal{F}_{t-1}] = \mathbb{E}[\sigma_t^2 z_t^2 | \mathcal{F}_{t-1}] = \sigma_t^2 \mathbb{E}[z_t^2] = \sigma_t^2
$$

since $z_t$ is independent of $\mathcal{F}_{t-1}$ and $\mathbb{E}[z_t^2]=1$. The unconditional variance, assuming stationarity, is obtained by taking unconditional expectations:

$$
\mathbb{E}[\sigma_t^2] = \alpha_0 + (\alpha_1 + \cdots + \alpha_q) \mathbb{E}[a_{t-1}^2]
$$

For stationarity, $\mathbb{E}[\sigma_t^2] = \mathbb{E}[a_t^2]$, so:

$$
\text{Var}(a_t) = \frac{\alpha_0}{1 - \sum_{i=1}^{q} \alpha_i}
$$

This requires $\sum_{i=1}^{q} \alpha_i < 1$ and $\alpha_i \geq 0$ for all $i$, $\alpha_0 > 0$.

### 4.2 GARCH(1,1) (Bollerslev, 1986)

The ARCH model requires many lags to capture persistent volatility. Bollerslev's Generalized ARCH model adds lagged conditional variance:

$$
\sigma_t^2 = \omega + \alpha a_{t-1}^2 + \beta \sigma_{t-1}^2
$$

where $\omega > 0$, $\alpha \geq 0$, $\beta \geq 0$.

**Stationarity conditions.** The GARCH(1,1) can be written as an ARCH($\infty$) by recursive substitution:

$$
\sigma_t^2 = \omega + \alpha a_{t-1}^2 + \beta(\omega + \alpha a_{t-2}^2 + \beta \sigma_{t-2}^2) = \frac{\omega}{1-\beta} + \alpha \sum_{j=0}^{\infty} \beta^j a_{t-1-j}^2
$$

(assuming $|\beta| < 1$). The unconditional variance is:

$$
\sigma^2 = \frac{\omega}{1 - \alpha - \beta}
$$

**Covariance stationarity** requires $\alpha + \beta < 1$. The quantity $\alpha + \beta$ is the **persistence** of volatility shocks. When $\alpha + \beta = 1$, we have an Integrated GARCH (IGARCH) where shocks to variance persist forever (the unconditional variance is infinite). Empirically, equity indices often exhibit $\alpha + \beta \approx 0.98$--$0.99$.

**Maximum Likelihood Estimation.** Assuming $z_t \sim N(0,1)$, the conditional log-likelihood for one observation is:

$$
\ell_t(\theta) = -\frac{1}{2} \log(2\pi) - \frac{1}{2} \log(\sigma_t^2) - \frac{1}{2} \frac{a_t^2}{\sigma_t^2}
$$

The full log-likelihood is $\mathcal{L}(\theta) = \sum_{t=1}^{T} \ell_t(\theta)$ where $\theta = (\omega, \alpha, \beta)$. The gradient for the GARCH(1,1) is computed via the chain rule. Define $h_t = \sigma_t^2$ for brevity:

$$
\frac{\partial \ell_t}{\partial \theta} = \frac{1}{2 h_t}\left(\frac{a_t^2}{h_t} - 1\right) \frac{\partial h_t}{\partial \theta}
$$

where the variance recursion gives:

$$
\frac{\partial h_t}{\partial \omega} = 1 + \beta \frac{\partial h_{t-1}}{\partial \omega}, \quad \frac{\partial h_t}{\partial \alpha} = a_{t-1}^2 + \beta \frac{\partial h_{t-1}}{\partial \alpha}, \quad \frac{\partial h_t}{\partial \beta} = h_{t-1} + \beta \frac{\partial h_{t-1}}{\partial \beta}
$$

These recursions are initialized at zero and computed forward. Optimization uses BFGS or L-BFGS-B with constraints $\omega > 0$, $\alpha \geq 0$, $\beta \geq 0$, $\alpha + \beta < 1$. Student-$t$ innovations are often used for heavier tails, replacing the Gaussian likelihood with:

$$
\ell_t = \log \Gamma\!\left(\frac{\nu+1}{2}\right) - \log \Gamma\!\left(\frac{\nu}{2}\right) - \frac{1}{2}\log[(\nu-2)\pi h_t] - \frac{\nu+1}{2}\log\!\left(1 + \frac{a_t^2}{(\nu-2)h_t}\right)
$$

### 4.3 EGARCH (Nelson, 1991)

The Exponential GARCH models the **log** of the conditional variance, eliminating the need for positivity constraints:

$$
\log(\sigma_t^2) = \omega + \alpha\left(\frac{|a_{t-1}|}{\sigma_{t-1}} - \sqrt{2/\pi}\right) + \gamma \frac{a_{t-1}}{\sigma_{t-1}} + \beta \log(\sigma_{t-1}^2)
$$

The term $\gamma a_{t-1}/\sigma_{t-1}$ captures the **leverage effect**: when $\gamma < 0$ (as typically estimated for equities), negative returns ($a_{t-1} < 0$) increase $\log(\sigma_t^2)$ more than positive returns of the same magnitude. The centering by $\sqrt{2/\pi} = \mathbb{E}[|z_t|]$ for standard normal $z_t$ ensures that $\mathbb{E}[\alpha(|z_t| - \sqrt{2/\pi})] = 0$, so the intercept $\omega$ retains a clean interpretation. Stationarity requires $|\beta| < 1$.

### 4.4 GJR-GARCH (Glosten-Jagannathan-Runkle, 1993)

The GJR-GARCH (also called Threshold GARCH) introduces asymmetry via an indicator function:

$$
\sigma_t^2 = \omega + (\alpha + \gamma \mathbb{1}_{a_{t-1}<0}) a_{t-1}^2 + \beta \sigma_{t-1}^2
$$

where $\mathbb{1}_{a_{t-1}<0}$ equals 1 when the past return shock is negative and 0 otherwise. The "news impact curve" shows that negative shocks have slope $\alpha + \gamma$ while positive shocks have slope $\alpha$, creating a V-shaped asymmetric response. The unconditional variance is:

$$
\sigma^2 = \frac{\omega}{1 - \alpha - \beta - \gamma/2}
$$

(the $\gamma/2$ arises because $\Pr(a_t < 0) = 1/2$ under symmetry).

### 4.5 DCC-GARCH (Engle, 2002)

For portfolios and risk management, we need the dynamic conditional correlation between multiple assets. The DCC-GARCH model separates univariate volatility modeling from correlation dynamics.

Let $\mathbf{r}_t$ be an $N$-vector of returns. The conditional covariance matrix is decomposed as:

$$
\mathbf{H}_t = \mathbf{D}_t \mathbf{R}_t \mathbf{D}_t
$$

where $\mathbf{D}_t = \text{diag}(\sigma_{1,t}, \ldots, \sigma_{N,t})$ contains univariate GARCH volatilities and $\mathbf{R}_t$ is the dynamic conditional correlation matrix.

**Step 1:** Fit univariate GARCH(1,1) to each series to obtain $\sigma_{i,t}$.

**Step 2:** Compute standardized residuals $\mathbf{u}_t = \mathbf{D}_t^{-1} \mathbf{a}_t$ and model the correlation dynamics:

$$
\mathbf{Q}_t = (1 - a - b) \bar{\mathbf{Q}} + a \, \mathbf{u}_{t-1} \mathbf{u}_{t-1}^\top + b \, \mathbf{Q}_{t-1}
$$

where $\bar{\mathbf{Q}}$ is the unconditional correlation matrix of $\mathbf{u}_t$, and $a, b \geq 0$, $a + b < 1$.

$$
\mathbf{R}_t = \text{diag}(\mathbf{Q}_t)^{-1/2} \, \mathbf{Q}_t \, \text{diag}(\mathbf{Q}_t)^{-1/2}
$$

This rescaling ensures $\mathbf{R}_t$ is a valid correlation matrix with ones on the diagonal. Estimation proceeds by two-stage quasi-maximum likelihood, which is computationally feasible even for large $N$.

---

## 5. Realized Volatility and High-Frequency Measures

With the availability of intraday tick data, model-free measures of volatility can be constructed directly from observed prices.

### 5.1 Realized Variance

Suppose we observe log-prices at $M+1$ equally spaced times within day $t$: $p_{t,0}, p_{t,1}, \ldots, p_{t,M}$. The intraday log-return over the $j$-th interval is $r_{t,j} = p_{t,j} - p_{t,j-1}$. The **realized variance** is:

$$
RV_t = \sum_{j=1}^{M} r_{t,j}^2
$$

Under the assumption that log-prices follow a continuous semimartingale $dp_t = \mu_t \, dt + \sigma_t \, dW_t$, as $M \to \infty$ (sampling frequency increases):

$$
RV_t \xrightarrow{p} IV_t = \int_0^1 \sigma_t^2(s) \, ds
$$

the **integrated variance**, which is the economically meaningful quantity. In practice, market microstructure noise (bid-ask bounce, discrete prices) causes $RV_t$ to be biased upward at very high frequencies. Solutions include subsampling (Zhang, Mykland, Ait-Sahalia), the two-scales estimator, and kernel-based estimators.

### 5.2 Bipower Variation

Bipower variation is designed to be robust to jumps:

$$
BV_t = \frac{\pi}{2} \sum_{j=2}^{M} |r_{t,j}| \cdot |r_{t,j-1}|
$$

The scaling factor $\mu_1^{-2} = \pi/2$ where $\mu_1 = \mathbb{E}[|Z|] = \sqrt{2/\pi}$ for $Z \sim N(0,1)$. Under the continuous semimartingale plus finite-activity jump model:

$$
dp_t = \mu_t \, dt + \sigma_t \, dW_t + \kappa_t \, dJ_t
$$

bipower variation converges to the integrated variance of the continuous component only:

$$
BV_t \xrightarrow{p} IV_t = \int_0^1 \sigma_t^2(s) \, ds
$$

while $RV_t \xrightarrow{p} IV_t + \sum_{j} \kappa_j^2$ (integrated variance plus jump contributions).

### 5.3 Jump Detection: BNS Test

The Barndorff-Nielsen and Shephard (2006) test exploits the difference between RV and BV to detect jumps. Under the null of no jumps, the test statistic is:

$$
Z_{\text{BNS}} = \frac{RV_t - BV_t}{\sqrt{\left(\frac{\pi^2}{4} + \pi - 5\right) \frac{1}{M} TQ_t}} \xrightarrow{d} N(0, 1)
$$

where $TQ_t$ is the **realized tripower quarticity**, an estimator of the integrated quarticity:

$$
TQ_t = M \cdot \mu_{4/3}^{-3} \sum_{j=3}^{M} |r_{t,j}|^{4/3} |r_{t,j-1}|^{4/3} |r_{t,j-2}|^{4/3}
$$

with $\mu_{4/3} = \mathbb{E}[|Z|^{4/3}] = 2^{2/3} \Gamma(7/6) / \Gamma(1/2)$. The jump variation component is then isolated as:

$$
J_t = \max(RV_t - BV_t, 0)
$$

$$
C_t = \min(RV_t, BV_t)
$$

where $C_t$ estimates the continuous component and $J_t$ estimates the jump component. This decomposition is essential for the HAR-RV-CJ model.

---

## 6. HAR-RV Model

### 6.1 Motivation and Specification

The Heterogeneous Autoregressive model of Realized Volatility (HAR-RV), proposed by Corsi (2009), captures the "cascade" of volatilities at different time horizons without requiring fractional integration. The model reflects the heterogeneous market hypothesis: different classes of traders (intraday, short-term, long-term) generate volatility persistence at different frequencies.

Define multi-period realized volatilities:

- **Daily:** $RV_t^{(d)} = RV_t$
- **Weekly:** $RV_t^{(w)} = \frac{1}{5} \sum_{j=0}^{4} RV_{t-j}$
- **Monthly:** $RV_t^{(m)} = \frac{1}{22} \sum_{j=0}^{21} RV_{t-j}$

The **HAR-RV** model is:

$$
RV_{t+1}^{(d)} = \beta_0 + \beta_d \, RV_t^{(d)} + \beta_w \, RV_t^{(w)} + \beta_m \, RV_t^{(m)} + \varepsilon_{t+1}
$$

### 6.2 Estimation and Properties

Despite its simplicity (OLS estimation of a linear regression), the HAR-RV model produces remarkably good out-of-sample volatility forecasts, often competitive with or superior to the GARCH family. The model generates long-memory-like behavior in the autocorrelation structure of $RV_t$ through the cascading structure of daily, weekly, and monthly components, even though each component is a short-memory AR(1) process.

The HAR-RV-CJ variant decomposes realized variance into continuous and jump components:

$$
RV_{t+1}^{(d)} = \beta_0 + \beta_{Cd} C_t^{(d)} + \beta_{Cw} C_t^{(w)} + \beta_{Cm} C_t^{(m)} + \beta_{Jd} J_t^{(d)} + \beta_{Jw} J_t^{(w)} + \beta_{Jm} J_t^{(m)} + \varepsilon_{t+1}
$$

Empirically, the continuous components are significantly more persistent and predictive than the jump components.

---

## 7. Cointegration and Error Correction

### 7.1 Spurious Regression

When two independent I(1) processes are regressed on each other, the $R^2$ does not converge to zero and the $t$-statistics diverge. This is the **spurious regression problem** (Granger and Newbold, 1974; Phillips, 1986). Cointegration provides the framework for legitimate inference with I(1) variables.

### 7.2 Cointegration Definition

Two (or more) I(1) processes $\{X_t\}$ and $\{Y_t\}$ are **cointegrated** if there exists a linear combination $Z_t = Y_t - \beta X_t$ that is I(0). The vector $[1, -\beta]$ is the **cointegrating vector**. Economically, cointegration implies a long-run equilibrium relationship: the series may wander individually, but they cannot drift arbitrarily far apart.

### 7.3 Engle-Granger Two-Step Procedure

**Step 1: Estimate the cointegrating regression.**

$$
Y_t = \alpha + \beta X_t + Z_t
$$

by OLS. The estimator $\hat{\beta}$ is **super-consistent**: it converges at rate $T$ (rather than the usual $\sqrt{T}$), though its distribution is non-standard and biased in finite samples (Stock, 1987).

**Step 2: Test the residuals for stationarity.** Compute $\hat{Z}_t = Y_t - \hat{\alpha} - \hat{\beta} X_t$ and apply the ADF test to $\hat{Z}_t$:

$$
\Delta \hat{Z}_t = \delta \hat{Z}_t + \sum_{j=1}^{p} \phi_j \Delta \hat{Z}_{t-j} + \varepsilon_t
$$

Reject $H_0$ of no cointegration if the ADF statistic is below the critical value. Note: the critical values are *not* the standard Dickey-Fuller values because $\hat{Z}_t$ uses estimated parameters. Special critical values (Engle-Granger, Phillips-Ouliaris) must be used.

**Step 3: Estimate the Error Correction Model (ECM).** If cointegration is confirmed:

$$
\Delta Y_t = \alpha_0 + \alpha_1 \hat{Z}_{t-1} + \sum_{j=1}^{p} \gamma_j \Delta Y_{t-j} + \sum_{j=0}^{q} \delta_j \Delta X_{t-j} + \varepsilon_t
$$

The error correction coefficient $\alpha_1 < 0$ governs the speed of adjustment back to equilibrium. The Granger Representation Theorem guarantees that cointegrated variables admit an ECM representation and vice versa.

### 7.4 Johansen Test

For systems with $K$ potentially cointegrated I(1) variables, the Johansen approach is based on the Vector Error Correction Model (VECM):

$$
\Delta \mathbf{Y}_t = \boldsymbol{\Pi} \mathbf{Y}_{t-1} + \sum_{j=1}^{p-1} \boldsymbol{\Gamma}_j \Delta \mathbf{Y}_{t-j} + \boldsymbol{\mu} + \boldsymbol{\varepsilon}_t
$$

where $\boldsymbol{\Pi} = \boldsymbol{\alpha} \boldsymbol{\beta}^\top$ is the **long-run impact matrix**. The rank of $\boldsymbol{\Pi}$ determines the number of cointegrating relationships:

- $\text{rank}(\boldsymbol{\Pi}) = 0$: no cointegration, all variables are I(1), use VAR in differences
- $\text{rank}(\boldsymbol{\Pi}) = r$ where $0 < r < K$: there are $r$ cointegrating vectors (columns of $\boldsymbol{\beta}$)
- $\text{rank}(\boldsymbol{\Pi}) = K$: all variables are stationary, use VAR in levels

**Derivation from VECM.** Start from the VAR($p$) in levels:

$$
\mathbf{Y}_t = \mathbf{A}_1 \mathbf{Y}_{t-1} + \mathbf{A}_2 \mathbf{Y}_{t-2} + \cdots + \mathbf{A}_p \mathbf{Y}_{t-p} + \boldsymbol{\mu} + \boldsymbol{\varepsilon}_t
$$

Subtract $\mathbf{Y}_{t-1}$ from both sides, then add and subtract terms to obtain:

$$
\Delta \mathbf{Y}_t = \boldsymbol{\Pi} \mathbf{Y}_{t-1} + \sum_{j=1}^{p-1} \boldsymbol{\Gamma}_j \Delta \mathbf{Y}_{t-j} + \boldsymbol{\mu} + \boldsymbol{\varepsilon}_t
$$

where $\boldsymbol{\Pi} = \sum_{i=1}^{p} \mathbf{A}_i - \mathbf{I}_K$ and $\boldsymbol{\Gamma}_j = -\sum_{i=j+1}^{p} \mathbf{A}_i$.

**Johansen's eigenvalue-based tests.** The test proceeds by reduced rank regression. After concentrating out the lagged differences and deterministic terms, we solve the eigenvalue problem:

$$
|S_{10} S_{00}^{-1} S_{01} - \lambda S_{11}| = 0
$$

where $S_{ij}$ are moment matrices from the residuals of auxiliary regressions. This yields $K$ ordered eigenvalues $\hat{\lambda}_1 \geq \hat{\lambda}_2 \geq \cdots \geq \hat{\lambda}_K$.

**Trace test statistic** (tests $H_0: \text{rank} \leq r$ vs $H_1: \text{rank} > r$):

$$
\Lambda_{\text{trace}}(r) = -T \sum_{i=r+1}^{K} \log(1 - \hat{\lambda}_i)
$$

**Maximum eigenvalue test statistic** (tests $H_0: \text{rank} = r$ vs $H_1: \text{rank} = r+1$):

$$
\Lambda_{\max}(r) = -T \log(1 - \hat{\lambda}_{r+1})
$$

Both statistics have non-standard distributions that depend on $K - r$ and the deterministic specification (e.g., whether a constant and/or trend is restricted to the cointegrating space). Critical values are obtained from Johansen's tables or by simulation.

---

## 8. Structural Breaks

Financial time series are often subject to regime changes: policy shifts, crises, market structure changes. Ignoring structural breaks leads to parameter instability, spurious unit roots (Perron, 1989), and misleading forecasts.

### 8.1 Chow Test

The Chow test examines whether the regression coefficients are stable across a known breakpoint $t^*$. Estimate the regression separately over $[1, t^*]$ and $[t^*+1, T]$, and jointly over $[1, T]$. The F-statistic is:

$$
F = \frac{(RSS_{\text{full}} - RSS_1 - RSS_2) / k}{(RSS_1 + RSS_2) / (T - 2k)}
$$

where $k$ is the number of parameters. Under $H_0$ of no break, $F \sim F(k, T - 2k)$.

### 8.2 Bai-Perron Test

When the break date is unknown and there may be multiple breaks, the Bai-Perron (1998, 2003) framework estimates break dates by minimizing the sum of squared residuals over all possible $m$-partition configurations. For $m$ breaks at $T_1, \ldots, T_m$:

$$
(\hat{T}_1, \ldots, \hat{T}_m) = \arg\min_{T_1, \ldots, T_m} \sum_{j=0}^{m} \sum_{t=T_j+1}^{T_{j+1}} (Y_t - \mathbf{X}_t^\top \boldsymbol{\beta}_j)^2
$$

The number of breaks is determined by sequential testing ($\text{SupF}$ tests) or the BIC. The global minimization is solved efficiently by dynamic programming (complexity $O(T^2)$ rather than the combinatorial $O(T^m)$).

### 8.3 CUSUM Test

The CUSUM (cumulative sum) test of Brown, Durbin, and Evans (1975) is based on the cumulative sum of recursive residuals:

$$
W_t = \frac{1}{\hat{\sigma}} \sum_{j=k+1}^{t} \hat{e}_j
$$

where $\hat{e}_j$ is the $j$-th recursive (one-step-ahead) residual and $\hat{\sigma}$ is its estimated standard deviation. Under $H_0$ of parameter stability, $W_t$ behaves like a Brownian bridge. The null is rejected if $W_t$ crosses the boundary lines $\pm c \sqrt{T - k}$, where $c$ depends on the significance level.

---

## 9. Long Memory Processes

### 9.1 Fractional Integration

Realized volatility and absolute returns often exhibit autocorrelations that decay hyperbolically ($\sim h^{2d-1}$ as $h \to \infty$) rather than geometrically, suggesting **long memory**. A process is fractionally integrated of order $d$, written $I(d)$, with $0 < d < 0.5$:

$$
(1 - L)^d X_t = \varepsilon_t
$$

where the fractional difference operator is defined via the binomial series:

$$
(1 - L)^d = \sum_{k=0}^{\infty} \binom{d}{k} (-L)^k = \sum_{k=0}^{\infty} \frac{\Gamma(k-d)}{\Gamma(-d)\Gamma(k+1)} L^k
$$

### 9.2 ARFIMA

The ARFIMA(p,d,q) model combines fractional integration with ARMA dynamics:

$$
\Phi(L)(1 - L)^d X_t = \Theta(L) \varepsilon_t
$$

where $d \in (-0.5, 0.5)$ for covariance stationarity and invertibility. For $0 < d < 0.5$, the spectral density diverges at the origin: $f(\lambda) \sim c |\lambda|^{-2d}$ as $\lambda \to 0$, capturing long-range dependence. Estimation methods include exact MLE (Sowell), Whittle likelihood (Fox-Taqqu), and semiparametric methods (GPH log-periodogram regression, local Whittle).

### 9.3 Hurst Exponent

The Hurst exponent $H$ quantifies the long-range dependence of a time series. For a fractionally integrated process, $H = d + 0.5$. Three regimes:

- $H = 0.5$: no long memory (short-range dependence)
- $0.5 < H < 1$: positive long-range dependence (persistence)
- $0 < H < 0.5$: anti-persistence (mean-reverting)

**Rescaled Range (R/S) Analysis.** For a series of length $n$, compute the range of cumulative deviations from the mean, scaled by the standard deviation:

$$
(R/S)_n = \frac{\max_{1 \leq k \leq n} \sum_{j=1}^{k}(X_j - \bar{X}_n) - \min_{1 \leq k \leq n} \sum_{j=1}^{k}(X_j - \bar{X}_n)}{s_n}
$$

Hurst's empirical law states $(R/S)_n \sim c \cdot n^H$, so $H$ is estimated from the slope of $\log(R/S)_n$ versus $\log n$.

**Detrended Fluctuation Analysis (DFA).** A more robust method that handles non-stationarity. Integrate the series, divide into blocks of size $n$, detrend each block by a polynomial fit, and compute the fluctuation function:

$$
F(n) = \sqrt{\frac{1}{N} \sum_{k=1}^{N} (X_k^{\text{int}} - X_k^{\text{trend}})^2}
$$

The scaling $F(n) \sim n^H$ provides the Hurst exponent.

---

## 10. Implementation: Python

```python
"""
Module 21: Time Series Analysis -- Python Implementation
Covers ADF test, ARIMA fitting, GARCH(1,1) MLE, Johansen cointegration, HAR-RV
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from typing import Tuple, Dict, Optional
import warnings

# =============================================================================
# 10.1 ADF Test from Scratch
# =============================================================================

def adf_test(y: np.ndarray, max_lags: int = None, criterion: str = 'aic') -> Dict:
    """
    Augmented Dickey-Fuller test for unit root.

    Parameters
    ----------
    y : np.ndarray
        Time series data (1D array).
    max_lags : int, optional
        Maximum number of lags to consider. Default: int(12*(T/100)^(1/4)).
    criterion : str
        Information criterion for lag selection: 'aic' or 'bic'.

    Returns
    -------
    dict with keys: 'statistic', 'p_value', 'lags', 'critical_values'
    """
    T = len(y)
    if max_lags is None:
        max_lags = int(12.0 * (T / 100.0) ** 0.25)

    dy = np.diff(y)
    y_lag = y[:-1]

    best_ic = np.inf
    best_lag = 0

    for p in range(0, max_lags + 1):
        n = len(dy) - p
        if n <= p + 2:
            continue

        # Construct regressors: constant, y_{t-1}, lagged differences
        X = np.column_stack([
            np.ones(n),
            y_lag[p:p + n]
        ])

        for j in range(1, p + 1):
            X = np.column_stack([X, dy[p - j:p - j + n]])

        Y = dy[p:p + n]

        # OLS estimation
        beta = np.linalg.lstsq(X, Y, rcond=None)[0]
        residuals = Y - X @ beta
        sse = np.sum(residuals ** 2)
        k = X.shape[1]

        if criterion == 'aic':
            ic = n * np.log(sse / n) + 2 * k
        else:  # bic
            ic = n * np.log(sse / n) + k * np.log(n)

        if ic < best_ic:
            best_ic = ic
            best_lag = p

    # Final regression with selected lag
    p = best_lag
    n = len(dy) - p
    X = np.column_stack([np.ones(n), y_lag[p:p + n]])
    for j in range(1, p + 1):
        X = np.column_stack([X, dy[p - j:p - j + n]])
    Y = dy[p:p + n]

    beta = np.linalg.lstsq(X, Y, rcond=None)[0]
    residuals = Y - X @ beta
    sigma2 = np.sum(residuals ** 2) / (n - X.shape[1])

    cov_beta = sigma2 * np.linalg.inv(X.T @ X)
    se_delta = np.sqrt(cov_beta[1, 1])

    adf_stat = beta[1] / se_delta

    # MacKinnon approximate p-values (model with constant, no trend)
    # Using interpolation of MacKinnon (1996) response surface
    tau_star = adf_stat
    if tau_star < -3.96:
        p_value = 0.001
    elif tau_star < -3.41:
        p_value = 0.01
    elif tau_star < -2.86:
        p_value = 0.05
    elif tau_star < -2.57:
        p_value = 0.10
    else:
        # Approximate using regression on MacKinnon surface
        p_value = min(1.0, np.exp(0.6 * tau_star + 1.5))

    return {
        'statistic': adf_stat,
        'p_value': p_value,
        'lags': best_lag,
        'critical_values': {'1%': -3.43, '5%': -2.86, '10%': -2.57}
    }


# =============================================================================
# 10.2 ARIMA Fitting
# =============================================================================

def arima_fit(y: np.ndarray, order: Tuple[int, int, int] = (1, 0, 0)) -> Dict:
    """
    Fit ARIMA(p,d,q) model via conditional MLE.

    Parameters
    ----------
    y : np.ndarray
        Time series data.
    order : tuple
        (p, d, q) order of the ARIMA model.

    Returns
    -------
    dict with keys: 'ar_params', 'ma_params', 'sigma2', 'aic', 'bic', 'residuals'
    """
    p, d, q = order

    # Apply differencing
    z = y.copy()
    for _ in range(d):
        z = np.diff(z)

    T = len(z)
    mean_z = np.mean(z)
    z_centered = z - mean_z

    def neg_log_likelihood(params):
        ar = params[:p]
        ma = params[p:p + q]
        sigma2 = params[p + q] ** 2  # Parameterize as sigma to avoid constraint

        if sigma2 < 1e-12:
            return 1e10

        # Check AR stationarity (roots outside unit circle)
        if p > 0:
            ar_poly = np.concatenate([[1], -ar])
            roots = np.roots(ar_poly)
            if np.any(np.abs(roots) <= 1.0):
                return 1e10

        # Compute residuals recursively (conditional likelihood)
        residuals = np.zeros(T)
        for t in range(T):
            pred = 0.0
            for j in range(p):
                if t - j - 1 >= 0:
                    pred += ar[j] * z_centered[t - j - 1]
            for j in range(q):
                if t - j - 1 >= 0:
                    pred += ma[j] * residuals[t - j - 1]
            residuals[t] = z_centered[t] - pred

        # Gaussian log-likelihood
        nll = 0.5 * T * np.log(2 * np.pi * sigma2) + 0.5 * np.sum(residuals ** 2) / sigma2
        return nll

    # Initial parameters
    x0 = np.zeros(p + q + 1)
    x0[-1] = np.std(z_centered)  # Initial sigma

    result = minimize(neg_log_likelihood, x0, method='Nelder-Mead',
                      options={'maxiter': 10000, 'xatol': 1e-8, 'fatol': 1e-8})

    params = result.x
    ar_params = params[:p]
    ma_params = params[p:p + q]
    sigma2 = params[p + q] ** 2

    k = p + q + 1  # Number of parameters (excl. mean)
    nll = result.fun
    aic = 2 * nll + 2 * k
    bic = 2 * nll + k * np.log(T)

    # Compute final residuals
    residuals = np.zeros(T)
    for t in range(T):
        pred = 0.0
        for j in range(p):
            if t - j - 1 >= 0:
                pred += ar_params[j] * z_centered[t - j - 1]
        for j in range(q):
            if t - j - 1 >= 0:
                pred += ma_params[j] * residuals[t - j - 1]
        residuals[t] = z_centered[t] - pred

    return {
        'ar_params': ar_params,
        'ma_params': ma_params,
        'sigma2': sigma2,
        'aic': aic,
        'bic': bic,
        'residuals': residuals,
        'mean': mean_z
    }


# =============================================================================
# 10.3 GARCH(1,1) MLE
# =============================================================================

class GARCH11:
    """
    GARCH(1,1) model: sigma_t^2 = omega + alpha * a_{t-1}^2 + beta * sigma_{t-1}^2
    Estimation via quasi-maximum likelihood with Gaussian innovations.
    """

    def __init__(self):
        self.omega = None
        self.alpha = None
        self.beta = None
        self.fitted_variance = None

    def _variance_recursion(self, params: np.ndarray, returns: np.ndarray) -> np.ndarray:
        """Compute conditional variance series given parameters."""
        omega, alpha, beta = params
        T = len(returns)
        h = np.zeros(T)

        # Initialize with unconditional variance
        h[0] = np.var(returns) if (alpha + beta) >= 1.0 else omega / (1.0 - alpha - beta)

        for t in range(1, T):
            h[t] = omega + alpha * returns[t - 1] ** 2 + beta * h[t - 1]
            h[t] = max(h[t], 1e-12)  # Numerical floor

        return h

    def _neg_log_likelihood(self, params: np.ndarray, returns: np.ndarray) -> float:
        """Negative Gaussian quasi-log-likelihood."""
        omega, alpha, beta = params

        # Parameter constraints
        if omega <= 0 or alpha < 0 or beta < 0:
            return 1e10
        if alpha + beta >= 1.0:
            return 1e10

        h = self._variance_recursion(params, returns)

        # Log-likelihood: sum of -0.5*(log(2*pi) + log(h_t) + r_t^2/h_t)
        ll = -0.5 * np.sum(np.log(2 * np.pi) + np.log(h) + returns ** 2 / h)
        return -ll

    def fit(self, returns: np.ndarray) -> Dict:
        """
        Fit GARCH(1,1) via MLE.

        Parameters
        ----------
        returns : np.ndarray
            Return series (assumed zero-mean or de-meaned).

        Returns
        -------
        dict with keys: 'omega', 'alpha', 'beta', 'persistence',
                        'unconditional_var', 'log_likelihood', 'aic', 'bic'
        """
        returns = returns - np.mean(returns)  # Demean
        T = len(returns)
        var_r = np.var(returns)

        # Initial values: omega=0.1*var, alpha=0.1, beta=0.8
        x0 = np.array([0.1 * var_r, 0.1, 0.8])

        # Bounds
        bounds = [(1e-8, 10 * var_r), (1e-8, 0.999), (1e-8, 0.999)]

        result = minimize(
            self._neg_log_likelihood, x0, args=(returns,),
            method='L-BFGS-B', bounds=bounds,
            options={'maxiter': 5000, 'ftol': 1e-12}
        )

        self.omega, self.alpha, self.beta = result.x
        self.fitted_variance = self._variance_recursion(result.x, returns)

        ll = -result.fun
        k = 3
        aic = -2 * ll + 2 * k
        bic = -2 * ll + k * np.log(T)

        return {
            'omega': self.omega,
            'alpha': self.alpha,
            'beta': self.beta,
            'persistence': self.alpha + self.beta,
            'unconditional_var': self.omega / (1 - self.alpha - self.beta),
            'log_likelihood': ll,
            'aic': aic,
            'bic': bic,
            'conditional_volatility': np.sqrt(self.fitted_variance)
        }

    def forecast(self, horizon: int = 10) -> np.ndarray:
        """
        Produce h-step-ahead variance forecasts.
        Closed-form: E[h_{t+k}] = sigma^2 + (alpha+beta)^k * (h_t - sigma^2)
        """
        if self.omega is None:
            raise ValueError("Model not fitted yet.")

        sigma2 = self.omega / (1 - self.alpha - self.beta)
        last_h = self.fitted_variance[-1]
        persistence = self.alpha + self.beta

        forecasts = np.array([
            sigma2 + persistence ** k * (last_h - sigma2)
            for k in range(1, horizon + 1)
        ])
        return forecasts


# =============================================================================
# 10.4 Johansen Cointegration Test
# =============================================================================

def johansen_test(data: np.ndarray, det_order: int = 0, k_ar_diff: int = 1) -> Dict:
    """
    Johansen cointegration test based on VECM.

    Parameters
    ----------
    data : np.ndarray
        (T x K) matrix of K time series.
    det_order : int
        Deterministic term: -1 (none), 0 (constant in coint), 1 (constant + trend).
    k_ar_diff : int
        Number of lagged differences in VECM.

    Returns
    -------
    dict with keys: 'eigenvalues', 'trace_stat', 'max_eigen_stat',
                    'trace_crit_95', 'eigenvectors'
    """
    T, K = data.shape

    # First differences
    dY = np.diff(data, axis=0)  # (T-1) x K
    Y_lag = data[:-1, :]         # (T-1) x K

    # Adjust for lagged differences
    n = T - 1 - k_ar_diff
    if n <= 0:
        raise ValueError("Insufficient data for the specified lag order.")

    dY_trimmed = dY[k_ar_diff:, :]
    Y_lag_trimmed = Y_lag[k_ar_diff:, :]

    # Construct lagged difference regressors
    Z = np.ones((n, 1)) if det_order >= 0 else np.empty((n, 0))
    for j in range(1, k_ar_diff + 1):
        Z = np.column_stack([Z, dY[k_ar_diff - j:k_ar_diff - j + n, :]])

    # Auxiliary regressions (concentrate out lagged differences)
    # Regress dY_trimmed on Z -> residuals R0
    # Regress Y_lag_trimmed on Z -> residuals R1
    if Z.shape[1] > 0:
        # R0 = dY_trimmed - Z * (Z'Z)^{-1} Z' dY_trimmed
        ZtZ_inv = np.linalg.inv(Z.T @ Z)
        M = np.eye(n) - Z @ ZtZ_inv @ Z.T
    else:
        M = np.eye(n)

    R0 = M @ dY_trimmed
    R1 = M @ Y_lag_trimmed

    # Moment matrices
    S00 = R0.T @ R0 / n
    S01 = R0.T @ R1 / n
    S10 = R1.T @ R0 / n
    S11 = R1.T @ R1 / n

    # Solve generalized eigenvalue problem: S10 S00^{-1} S01 v = lambda S11 v
    S00_inv = np.linalg.inv(S00)
    S11_inv = np.linalg.inv(S11)

    A = S11_inv @ S10 @ S00_inv @ S01
    eigenvalues, eigenvectors = np.linalg.eig(A)

    # Sort in descending order
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.real(eigenvalues[idx])
    eigenvectors = np.real(eigenvectors[:, idx])

    # Clip eigenvalues to [0, 1)
    eigenvalues = np.clip(eigenvalues, 0, 0.9999)

    # Trace statistics
    trace_stat = np.zeros(K)
    for r in range(K):
        trace_stat[r] = -n * np.sum(np.log(1 - eigenvalues[r:]))

    # Max eigenvalue statistics
    max_eigen_stat = np.zeros(K)
    for r in range(K):
        max_eigen_stat[r] = -n * np.log(1 - eigenvalues[r])

    # Approximate 95% critical values (Osterwald-Lenum tables for det_order=0)
    # These are for K=2; for general K, use proper tables
    trace_crit_95 = {
        2: [15.41, 3.76],
        3: [29.68, 15.41, 3.76],
        4: [47.21, 29.68, 15.41, 3.76]
    }

    return {
        'eigenvalues': eigenvalues,
        'trace_stat': trace_stat,
        'max_eigen_stat': max_eigen_stat,
        'trace_crit_95': trace_crit_95.get(K, None),
        'eigenvectors': eigenvectors,
        'n_obs': n
    }


# =============================================================================
# 10.5 HAR-RV Model
# =============================================================================

def har_rv_estimate(rv_daily: np.ndarray) -> Dict:
    """
    Estimate HAR-RV model:
    RV_{t+1} = beta_0 + beta_d * RV_t^(d) + beta_w * RV_t^(w) + beta_m * RV_t^(m) + eps

    Parameters
    ----------
    rv_daily : np.ndarray
        Daily realized variance series.

    Returns
    -------
    dict with keys: 'beta', 'se', 't_stats', 'r_squared', 'residuals'
    """
    T = len(rv_daily)
    if T < 23:
        raise ValueError("Need at least 23 observations for HAR-RV.")

    # Construct weekly and monthly RV
    rv_w = np.array([np.mean(rv_daily[max(0, t - 4):t + 1]) for t in range(T)])
    rv_m = np.array([np.mean(rv_daily[max(0, t - 21):t + 1]) for t in range(T)])

    # Align: use t = 21,...,T-2 for estimation (need 22 lags for monthly)
    start = 21
    n = T - start - 1

    Y = rv_daily[start + 1:start + 1 + n]  # RV_{t+1}
    X = np.column_stack([
        np.ones(n),
        rv_daily[start:start + n],    # RV_t^(d)
        rv_w[start:start + n],         # RV_t^(w)
        rv_m[start:start + n]          # RV_t^(m)
    ])

    # OLS
    beta = np.linalg.lstsq(X, Y, rcond=None)[0]
    residuals = Y - X @ beta
    sigma2 = np.sum(residuals ** 2) / (n - 4)

    cov_beta = sigma2 * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(cov_beta))
    t_stats = beta / se

    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((Y - np.mean(Y)) ** 2)
    r_squared = 1 - ss_res / ss_tot

    return {
        'beta': beta,
        'se': se,
        't_stats': t_stats,
        'r_squared': r_squared,
        'residuals': residuals,
        'labels': ['const', 'RV_daily', 'RV_weekly', 'RV_monthly']
    }


# =============================================================================
# Usage Example
# =============================================================================

if __name__ == "__main__":
    np.random.seed(42)
    T = 1000

    # --- ADF Test ---
    # Generate random walk (unit root)
    rw = np.cumsum(np.random.randn(T))
    result = adf_test(rw)
    print("=== ADF Test (Random Walk) ===")
    print(f"  Statistic: {result['statistic']:.4f}")
    print(f"  p-value:   {result['p_value']:.4f}")
    print(f"  Lags:      {result['lags']}")
    print(f"  Critical:  {result['critical_values']}")

    # --- ARIMA Fit ---
    ar1 = np.zeros(T)
    for t in range(1, T):
        ar1[t] = 0.7 * ar1[t - 1] + np.random.randn()

    result_arima = arima_fit(ar1, order=(1, 0, 0))
    print("\n=== ARIMA(1,0,0) Fit ===")
    print(f"  AR(1) coeff: {result_arima['ar_params'][0]:.4f} (true: 0.7)")
    print(f"  Sigma^2:     {result_arima['sigma2']:.4f}")
    print(f"  AIC:         {result_arima['aic']:.2f}")

    # --- GARCH(1,1) ---
    # Simulate GARCH(1,1)
    omega_true, alpha_true, beta_true = 0.05, 0.10, 0.85
    returns = np.zeros(T)
    h = np.zeros(T)
    h[0] = omega_true / (1 - alpha_true - beta_true)
    for t in range(1, T):
        h[t] = omega_true + alpha_true * returns[t - 1] ** 2 + beta_true * h[t - 1]
        returns[t] = np.sqrt(h[t]) * np.random.randn()

    garch = GARCH11()
    result_garch = garch.fit(returns)
    print("\n=== GARCH(1,1) MLE ===")
    print(f"  omega: {result_garch['omega']:.6f} (true: {omega_true})")
    print(f"  alpha: {result_garch['alpha']:.4f} (true: {alpha_true})")
    print(f"  beta:  {result_garch['beta']:.4f} (true: {beta_true})")
    print(f"  Persistence: {result_garch['persistence']:.4f}")

    # --- Johansen Test ---
    # Cointegrated system: Y1 = random walk, Y2 = Y1 + stationary noise
    rw1 = np.cumsum(np.random.randn(T))
    rw2 = rw1 + 0.5 * np.random.randn(T)
    data = np.column_stack([rw1, rw2])

    result_joh = johansen_test(data, det_order=0, k_ar_diff=1)
    print("\n=== Johansen Cointegration Test ===")
    print(f"  Eigenvalues: {result_joh['eigenvalues']}")
    print(f"  Trace stats: {result_joh['trace_stat']}")
    print(f"  95% critical: {result_joh['trace_crit_95']}")

    # --- HAR-RV ---
    rv = np.exp(np.random.randn(500) * 0.5) * 0.0001
    result_har = har_rv_estimate(rv)
    print("\n=== HAR-RV Model ===")
    for label, b, se, t in zip(result_har['labels'], result_har['beta'],
                                result_har['se'], result_har['t_stats']):
        print(f"  {label:12s}: beta={b:.6f}, se={se:.6f}, t={t:.3f}")
    print(f"  R-squared: {result_har['r_squared']:.4f}")
```

---

## 11. Implementation: C++

```cpp
/**
 * Module 21: High-Performance GARCH(1,1) Estimation in C++
 *
 * Compile: g++ -O3 -std=c++17 -o garch garch.cpp -lm
 *
 * Features:
 * - Numerically stable variance recursion with floor
 * - Analytic gradient for L-BFGS optimization
 * - Forecast generation with closed-form multi-step ahead
 */

#include <cmath>
#include <vector>
#include <numeric>
#include <algorithm>
#include <stdexcept>
#include <iostream>
#include <iomanip>
#include <limits>

// ---------------------------------------------------------------------------
// Simple L-BFGS-B optimizer (simplified for self-contained module)
// In production, link against dlib, NLopt, or Eigen's unsupported modules
// ---------------------------------------------------------------------------

struct OptimResult {
    std::vector<double> x;
    double fval;
    int iterations;
    bool converged;
};

// Nelder-Mead simplex (fallback optimizer for constrained problems)
OptimResult nelder_mead(
    std::function<double(const std::vector<double>&)> func,
    std::vector<double> x0,
    int max_iter = 10000,
    double tol = 1e-10
) {
    const int n = static_cast<int>(x0.size());
    const double alpha = 1.0, gamma_coeff = 2.0, rho = 0.5, sigma = 0.5;

    // Initialize simplex
    std::vector<std::vector<double>> simplex(n + 1, x0);
    for (int i = 0; i < n; ++i) {
        simplex[i + 1][i] += (x0[i] != 0.0) ? 0.05 * std::abs(x0[i]) : 0.00025;
    }

    std::vector<double> fvals(n + 1);
    for (int i = 0; i <= n; ++i) {
        fvals[i] = func(simplex[i]);
    }

    int iter = 0;
    for (; iter < max_iter; ++iter) {
        // Sort vertices by function value
        std::vector<int> idx(n + 1);
        std::iota(idx.begin(), idx.end(), 0);
        std::sort(idx.begin(), idx.end(),
                  [&](int a, int b) { return fvals[a] < fvals[b]; });

        // Check convergence
        double spread = fvals[idx[n]] - fvals[idx[0]];
        if (spread < tol) break;

        // Centroid of best n vertices
        std::vector<double> centroid(n, 0.0);
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                centroid[j] += simplex[idx[i]][j];
            }
        }
        for (int j = 0; j < n; ++j) centroid[j] /= n;

        // Reflection
        std::vector<double> xr(n);
        for (int j = 0; j < n; ++j)
            xr[j] = centroid[j] + alpha * (centroid[j] - simplex[idx[n]][j]);
        double fr = func(xr);

        if (fr < fvals[idx[0]]) {
            // Expansion
            std::vector<double> xe(n);
            for (int j = 0; j < n; ++j)
                xe[j] = centroid[j] + gamma_coeff * (xr[j] - centroid[j]);
            double fe = func(xe);
            if (fe < fr) {
                simplex[idx[n]] = xe;
                fvals[idx[n]] = fe;
            } else {
                simplex[idx[n]] = xr;
                fvals[idx[n]] = fr;
            }
        } else if (fr < fvals[idx[n - 1]]) {
            simplex[idx[n]] = xr;
            fvals[idx[n]] = fr;
        } else {
            // Contraction
            std::vector<double> xc(n);
            for (int j = 0; j < n; ++j)
                xc[j] = centroid[j] + rho * (simplex[idx[n]][j] - centroid[j]);
            double fc = func(xc);
            if (fc < fvals[idx[n]]) {
                simplex[idx[n]] = xc;
                fvals[idx[n]] = fc;
            } else {
                // Shrink
                for (int i = 1; i <= n; ++i) {
                    for (int j = 0; j < n; ++j) {
                        simplex[idx[i]][j] = simplex[idx[0]][j]
                            + sigma * (simplex[idx[i]][j] - simplex[idx[0]][j]);
                    }
                    fvals[idx[i]] = func(simplex[idx[i]]);
                }
            }
        }
    }

    // Find best
    int best = 0;
    for (int i = 1; i <= n; ++i) {
        if (fvals[i] < fvals[best]) best = i;
    }

    return {simplex[best], fvals[best], iter, iter < max_iter};
}

// ---------------------------------------------------------------------------
// GARCH(1,1) Engine
// ---------------------------------------------------------------------------

class GARCH11 {
public:
    struct Params {
        double omega;
        double alpha;
        double beta;
    };

    struct FitResult {
        Params params;
        double persistence;
        double unconditional_var;
        double log_likelihood;
        double aic;
        double bic;
        std::vector<double> conditional_variance;
        int iterations;
    };

private:
    std::vector<double> returns_;
    int T_;

    // Compute conditional variance series
    std::vector<double> variance_recursion(const Params& p) const {
        std::vector<double> h(T_);
        double uncond = (p.alpha + p.beta < 1.0) ?
                        p.omega / (1.0 - p.alpha - p.beta) :
                        sample_variance();

        h[0] = uncond;
        for (int t = 1; t < T_; ++t) {
            h[t] = p.omega + p.alpha * returns_[t - 1] * returns_[t - 1]
                   + p.beta * h[t - 1];
            if (h[t] < 1e-14) h[t] = 1e-14;  // Numerical floor
        }
        return h;
    }

    double sample_variance() const {
        double mean = std::accumulate(returns_.begin(), returns_.end(), 0.0) / T_;
        double var = 0.0;
        for (double r : returns_) var += (r - mean) * (r - mean);
        return var / T_;
    }

    // Negative log-likelihood (Gaussian quasi-MLE)
    double neg_log_likelihood(const Params& p) const {
        if (p.omega <= 0 || p.alpha < 0 || p.beta < 0) return 1e15;
        if (p.alpha + p.beta >= 1.0) return 1e15;

        auto h = variance_recursion(p);

        double nll = 0.0;
        constexpr double log2pi = 1.8378770664093453;
        for (int t = 0; t < T_; ++t) {
            nll += log2pi + std::log(h[t]) + returns_[t] * returns_[t] / h[t];
        }
        return 0.5 * nll;
    }

public:
    explicit GARCH11(const std::vector<double>& returns)
        : returns_(returns), T_(static_cast<int>(returns.size()))
    {
        if (T_ < 10) throw std::invalid_argument("Need at least 10 observations.");

        // Demean
        double mean = std::accumulate(returns_.begin(), returns_.end(), 0.0) / T_;
        for (auto& r : returns_) r -= mean;
    }

    FitResult fit() {
        double sv = sample_variance();

        // Objective: map unconstrained params to constrained space
        // Use log/logit transforms for numerical stability
        auto objective = [this](const std::vector<double>& x) -> double {
            double omega = std::exp(x[0]);
            double alpha = 0.5 / (1.0 + std::exp(-x[1]));  // (0, 0.5)
            double beta  = 0.998 / (1.0 + std::exp(-x[2])); // (0, 0.998)

            if (alpha + beta >= 0.9999) return 1e15;

            return neg_log_likelihood({omega, alpha, beta});
        };

        // Initial values
        std::vector<double> x0 = {
            std::log(0.05 * sv),  // omega ~ 5% of sample variance
            0.0,                   // alpha ~ 0.25
            2.0                    // beta ~ 0.88
        };

        auto result = nelder_mead(objective, x0, 20000, 1e-12);

        // Transform back
        Params p;
        p.omega = std::exp(result.x[0]);
        p.alpha = 0.5 / (1.0 + std::exp(-result.x[1]));
        p.beta  = 0.998 / (1.0 + std::exp(-result.x[2]));

        auto h = variance_recursion(p);
        double ll = -neg_log_likelihood(p);

        FitResult fr;
        fr.params = p;
        fr.persistence = p.alpha + p.beta;
        fr.unconditional_var = p.omega / (1.0 - p.alpha - p.beta);
        fr.log_likelihood = ll;
        fr.aic = -2.0 * ll + 2.0 * 3;
        fr.bic = -2.0 * ll + 3.0 * std::log(static_cast<double>(T_));
        fr.conditional_variance = h;
        fr.iterations = result.iterations;

        return fr;
    }

    // Multi-step-ahead forecast (closed-form)
    std::vector<double> forecast(const FitResult& fr, int horizon) const {
        double sigma2 = fr.unconditional_var;
        double last_h = fr.conditional_variance.back();
        double pers = fr.persistence;

        std::vector<double> fcast(horizon);
        for (int k = 0; k < horizon; ++k) {
            fcast[k] = sigma2 + std::pow(pers, k + 1) * (last_h - sigma2);
        }
        return fcast;
    }
};

// ---------------------------------------------------------------------------
// Main: demonstration with simulated data
// ---------------------------------------------------------------------------

int main() {
    // Simulate GARCH(1,1) process
    const int T = 5000;
    const double omega_true = 0.02, alpha_true = 0.09, beta_true = 0.89;

    std::vector<double> returns(T, 0.0);
    std::vector<double> h(T, 0.0);

    // Simple LCG for reproducibility (replace with <random> in production)
    auto randn = [](unsigned long& seed) -> double {
        // Box-Muller from uniform LCG
        seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
        double u1 = (seed >> 11) * (1.0 / 9007199254740992.0);
        seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
        double u2 = (seed >> 11) * (1.0 / 9007199254740992.0);
        u1 = std::max(u1, 1e-15);
        return std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * M_PI * u2);
    };

    unsigned long seed = 42;
    h[0] = omega_true / (1.0 - alpha_true - beta_true);
    returns[0] = std::sqrt(h[0]) * randn(seed);

    for (int t = 1; t < T; ++t) {
        h[t] = omega_true + alpha_true * returns[t - 1] * returns[t - 1]
               + beta_true * h[t - 1];
        returns[t] = std::sqrt(h[t]) * randn(seed);
    }

    // Fit GARCH(1,1)
    GARCH11 model(returns);
    auto result = model.fit();

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "=== GARCH(1,1) Estimation (C++) ===" << std::endl;
    std::cout << "  omega:       " << result.params.omega
              << "  (true: " << omega_true << ")" << std::endl;
    std::cout << "  alpha:       " << result.params.alpha
              << "  (true: " << alpha_true << ")" << std::endl;
    std::cout << "  beta:        " << result.params.beta
              << "  (true: " << beta_true << ")" << std::endl;
    std::cout << "  persistence: " << result.persistence
              << "  (true: " << alpha_true + beta_true << ")" << std::endl;
    std::cout << "  LogLik:      " << result.log_likelihood << std::endl;
    std::cout << "  AIC:         " << result.aic << std::endl;
    std::cout << "  BIC:         " << result.bic << std::endl;
    std::cout << "  Iterations:  " << result.iterations << std::endl;

    // Forecast
    auto fcast = model.forecast(result, 10);
    std::cout << "\n  Variance Forecasts:" << std::endl;
    for (int k = 0; k < 10; ++k) {
        std::cout << "    h[t+" << k + 1 << "] = " << fcast[k]
                  << "  (vol = " << std::sqrt(fcast[k]) << ")" << std::endl;
    }

    return 0;
}
```

---

## 12. Exercises

### Conceptual

**Exercise 21.1.** *Stationarity and differencing.*
An analyst fits an ARMA(2,1) model to the log-price of a stock. The residuals show strong autocorrelation at all lags, and the ADF test on log-prices fails to reject the null. (a) Explain what went wrong. (b) What is the correct modeling procedure? (c) Why is it dangerous to difference a stationary series?

**Exercise 21.2.** *GARCH stationarity.*
For a GARCH(1,1) model with $\omega = 0.00001$, $\alpha = 0.15$, $\beta = 0.83$, (a) compute the unconditional variance and annualized volatility (assuming 252 trading days); (b) compute the half-life of a volatility shock; (c) is this model covariance stationary? Why or why not?

**Exercise 21.3.** *Leverage effect.*
Explain, with reference to both the EGARCH and GJR-GARCH specifications, how each model captures the empirical observation that stock market volatility increases more after negative returns than after positive returns of the same magnitude. Which model is more parsimonious in terms of parameter constraints?

### Computational

**Exercise 21.4.** *ADF power study.*
Simulate 1000 paths of length $T = 200$ from: (a) a random walk, (b) an AR(1) with $\phi = 0.95$, (c) an AR(1) with $\phi = 0.99$. Apply the ADF test at 5% significance to each. Report the empirical rejection rates. Discuss the difficulty of distinguishing near-unit-root stationary processes from true unit roots.

**Exercise 21.5.** *GARCH MLE.*
Using daily S&P 500 returns (or simulated GARCH data): (a) Estimate GARCH(1,1) with both Gaussian and Student-$t$ innovations. (b) Compare the log-likelihoods and BIC values. (c) Plot the news impact curves for both. (d) Compute 10-day-ahead variance forecasts.

**Exercise 21.6.** *Pairs trading with cointegration.*
Select two cointegrated stocks (or simulate a cointegrated pair). (a) Verify cointegration using both Engle-Granger and Johansen tests. (b) Estimate the VECM. (c) Implement a trading strategy that goes long the undervalued stock and short the overvalued stock when the spread deviates by more than 2 standard deviations. (d) Backtest and report the Sharpe ratio.

**Exercise 21.7.** *HAR-RV forecasting.*
Using 5-minute return data for a major equity index: (a) Compute daily realized variance. (b) Apply the BNS jump test and decompose RV into continuous and jump components. (c) Estimate both HAR-RV and HAR-RV-CJ models. (d) Compare out-of-sample forecasting performance using RMSE and QLIKE loss functions.

**Exercise 21.8.** *Long memory in volatility.*
(a) Compute the Hurst exponent for daily absolute returns of a stock index using both R/S analysis and DFA. (b) Fit an ARFIMA(1,$d$,1) model to the log realized variance series. (c) Compare the autocorrelation structure implied by the ARFIMA model with that of the HAR-RV model.

---

## 13. Summary and Concept Map

This module developed the core time series toolkit for quantitative finance, from the fundamental concept of stationarity through linear ARMA models, volatility dynamics, high-frequency measures, long-run equilibrium relationships, and structural instability.

**Key takeaways:**

- Stationarity (tested via ADF, KPSS, PP) determines whether to model in levels or differences. Financial prices are typically I(1); returns are typically I(0).
- ARMA models capture linear dynamics in stationary series. The Box-Jenkins approach uses ACF/PACF patterns for identification.
- The GARCH family (ARCH, GARCH, EGARCH, GJR-GARCH, DCC-GARCH) models the volatility clustering and leverage effects that are universal in financial returns.
- Realized volatility provides model-free estimates from high-frequency data. The HAR-RV model captures long-memory-like behavior with a simple linear structure.
- Cointegration (Engle-Granger, Johansen) formalizes long-run equilibrium relationships between I(1) series, with applications to pairs trading and spread modeling.
- Structural breaks and long memory extend the toolkit to handle non-constant parameters and hyperbolic memory decay.

```mermaid
graph TD
    A[Stationarity Testing<br>ADF / KPSS / PP] --> B{Stationary?}
    B -->|Yes| C[ARMA Modeling<br>Box-Jenkins]
    B -->|No, I(1)| D[Differencing<br>ARIMA / SARIMA]
    B -->|Fractional| E[ARFIMA<br>Long Memory]

    C --> F[Volatility Modeling]
    D --> F

    F --> G[ARCH / GARCH<br>Conditional Variance]
    F --> H[EGARCH / GJR-GARCH<br>Asymmetry & Leverage]
    F --> I[DCC-GARCH<br>Multivariate Correlation]

    D --> J[Cointegration<br>Engle-Granger / Johansen]
    J --> K[VECM<br>Error Correction]
    K --> L[Pairs Trading<br>Spread Models]

    F --> M[High-Frequency Data]
    M --> N[Realized Variance<br>Bipower Variation]
    N --> O[Jump Detection<br>BNS Test]
    N --> P[HAR-RV Model<br>Multi-scale Forecasting]

    A --> Q[Structural Breaks<br>Chow / Bai-Perron / CUSUM]
    E --> R[Hurst Exponent<br>R/S / DFA]

    G --> S[MLE Estimation]
    H --> S
    I --> S

    style A fill:#1a1a2e,stroke:#e94560,color:#fff
    style F fill:#1a1a2e,stroke:#e94560,color:#fff
    style J fill:#1a1a2e,stroke:#e94560,color:#fff
    style M fill:#1a1a2e,stroke:#e94560,color:#fff
    style S fill:#0f3460,stroke:#e94560,color:#fff
```

---

*Next: [Module 22 — Kalman Filters & State-Space Models](../asset-pricing/22_kalman_filters.md)*

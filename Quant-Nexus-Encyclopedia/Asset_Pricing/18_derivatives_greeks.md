# Module 18: Derivatives Pricing & the Greeks

> **Prerequisites:** Module 04 (Probability & Stochastic Processes), Module 05 (Ito Calculus & SDEs), Module 08 (Fundamental Theorems of Asset Pricing)
>
> **Builds toward:** Module 19 (Local & Stochastic Volatility Models), Module 20 (Exotic Derivatives & Structured Products), Module 23 (Monte Carlo Methods), Module 30 (Algorithmic Greeks & AAD)

---

## Table of Contents

1. [The Binomial Model](#1-the-binomial-model)
2. [Black-Scholes via Replication and PDE](#2-black-scholes-via-replication-and-pde)
3. [Black-Scholes via Risk-Neutral Pricing](#3-black-scholes-via-risk-neutral-pricing)
4. [Black-Scholes Closed-Form Formulae](#4-black-scholes-closed-form-formulae)
5. [The Greeks: Complete Taxonomy](#5-the-greeks-complete-taxonomy)
6. [Implied Volatility](#6-implied-volatility)
7. [Volatility Smile and Skew](#7-volatility-smile-and-skew)
8. [Exotic Options](#8-exotic-options)
9. [American Options and Early Exercise](#9-american-options-and-early-exercise)
10. [Put-Call Parity, Symmetry, and Static Replication](#10-put-call-parity-symmetry-and-static-replication)
11. [Python Implementation](#11-python-implementation)
12. [C++ Implementation](#12-c-implementation)
13. [Exercises](#13-exercises)
14. [Summary and Bridge to Module 19](#14-summary-and-bridge-to-module-19)

---

## 1. The Binomial Model

The binomial model is the simplest complete market model and provides the conceptual foundation for every result that follows. We build it from first principles.

### 1.1 One-Period Model

Consider a single period $[0, T]$. A stock has price $S_0$ at time 0 and can move to one of two states:

$$S_T = \begin{cases} S_u = u \cdot S_0 & \text{with probability } p \\ S_d = d \cdot S_0 & \text{with probability } 1 - p \end{cases}$$

where $u > 1 > d > 0$. A risk-free bond grows from $B_0 = 1$ to $B_T = 1 + r$ (or $e^{rT}$ in continuous compounding). We assume the no-arbitrage condition:

$$d < e^{rT} < u$$

**Deriving the risk-neutral probability.** Consider a derivative with payoff $V_u$ in the up state and $V_d$ in the down state. We form a replicating portfolio of $\Delta$ shares of stock and $\psi$ units of bond:

$$\Delta \cdot u S_0 + \psi \cdot e^{rT} = V_u$$
$$\Delta \cdot d S_0 + \psi \cdot e^{rT} = V_d$$

Solving for $\Delta$ (the hedge ratio):

$$\Delta = \frac{V_u - V_d}{(u - d) S_0}$$

By no-arbitrage, the derivative price must equal the cost of the replicating portfolio:

$$V_0 = \Delta \cdot S_0 + \psi = e^{-rT}\bigl[q \, V_u + (1 - q) \, V_d\bigr]$$

where the **risk-neutral probability** is:

$$\boxed{q = \frac{e^{rT} - d}{u - d}}$$

The condition $d < e^{rT} < u$ guarantees $q \in (0, 1)$. Under this measure $\mathbb{Q}$, the discounted stock price is a martingale:

$$S_0 = e^{-rT} \mathbb{E}^{\mathbb{Q}}[S_T] = e^{-rT}[q \cdot u S_0 + (1-q) \cdot d S_0]$$

### 1.2 Multi-Period CRR Tree

Cox, Ross, and Rubinstein (1979) extended this to $n$ periods, each of length $\delta t = T/n$. Their parameter choices ensure the tree recombines and matches the first two moments of the log-normal distribution:

$$u = e^{\sigma\sqrt{\delta t}}, \qquad d = e^{-\sigma\sqrt{\delta t}} = \frac{1}{u}$$

$$q = \frac{e^{r\,\delta t} - d}{u - d}$$

After $n$ steps, the stock can reach $n + 1$ distinct nodes. At step $j$ (having gone up $k$ times out of $j$ steps):

$$S_{j,k} = S_0 \, u^k \, d^{j-k}$$

The European option price is computed by backward induction or directly via:

$$V_0 = e^{-rT} \sum_{k=0}^{n} \binom{n}{k} q^k (1-q)^{n-k} \, \text{Payoff}(S_{n,k})$$

```mermaid
graph TD
    S0["S₀"] --> Su["S₀u"]
    S0 --> Sd["S₀d"]
    Su --> Suu["S₀u²"]
    Su --> Sud["S₀ud"]
    Sd --> Sud2["S₀ud"]
    Sd --> Sdd["S₀d²"]
    Suu --> Suuu["S₀u³"]
    Suu --> Suud["S₀u²d"]
    Sud --> Suud2["S₀u²d"]
    Sud --> Sudd["S₀ud²"]
    Sud2 --> Suud3["S₀u²d"]
    Sud2 --> Sudd2["S₀ud²"]
    Sdd --> Sudd3["S₀ud²"]
    Sdd --> Sddd["S₀d³"]

    style S0 fill:#1a73e8,color:#fff
    style Su fill:#34a853,color:#fff
    style Sd fill:#ea4335,color:#fff
```

### 1.3 Convergence to Black-Scholes (Proof Sketch)

**Theorem.** As $n \to \infty$, the CRR binomial price converges to the Black-Scholes price.

*Proof sketch.* Define $X_n = \sum_{i=1}^{n} Y_i$ where $Y_i = +1$ (up) with probability $q$ and $Y_i = -1$ (down) with probability $1 - q$. Then:

$$\ln\frac{S_T}{S_0} = X_n \sigma\sqrt{\delta t} = \sigma\sqrt{\delta t}\sum_{i=1}^{n} Y_i$$

Compute the moments under $\mathbb{Q}$:

$$\mathbb{E}^{\mathbb{Q}}[Y_i] = 2q - 1 = \frac{e^{r\delta t} - d}{u - d} \cdot 2 - 1$$

Expanding $e^{r\delta t} \approx 1 + r\delta t$, $u \approx 1 + \sigma\sqrt{\delta t} + \frac{1}{2}\sigma^2 \delta t$, $d \approx 1 - \sigma\sqrt{\delta t} + \frac{1}{2}\sigma^2 \delta t$, we obtain:

$$\mathbb{E}^{\mathbb{Q}}[Y_i] \approx \frac{r - \frac{1}{2}\sigma^2}{\sigma}\sqrt{\delta t}$$

$$\text{Var}^{\mathbb{Q}}(Y_i) = 1 - (\mathbb{E}^{\mathbb{Q}}[Y_i])^2 \approx 1 - O(\delta t)$$

Therefore:

$$\mathbb{E}^{\mathbb{Q}}\!\left[\ln\frac{S_T}{S_0}\right] = n \cdot \sigma\sqrt{\delta t} \cdot \frac{r - \frac{1}{2}\sigma^2}{\sigma}\sqrt{\delta t} = \left(r - \tfrac{1}{2}\sigma^2\right)T$$

$$\text{Var}^{\mathbb{Q}}\!\left(\ln\frac{S_T}{S_0}\right) = n \cdot \sigma^2 \delta t \cdot \text{Var}(Y_i) \to \sigma^2 T$$

By the Central Limit Theorem:

$$\ln\frac{S_T}{S_0} \xrightarrow{d} \mathcal{N}\!\left(\left(r - \tfrac{1}{2}\sigma^2\right)T, \; \sigma^2 T\right)$$

which is precisely the distribution of $S_T$ under $\mathbb{Q}$ in the Black-Scholes model. Since the payoff function for a European option is bounded and continuous a.e., convergence in distribution implies convergence of the expected discounted payoff. $\blacksquare$

---

## 2. Black-Scholes via Replication and PDE

### 2.1 Setup and Assumptions

The Black-Scholes model assumes:
- The stock follows geometric Brownian motion: $dS_t = \mu S_t \, dt + \sigma S_t \, dW_t$
- Continuous trading with no transaction costs or taxes
- Constant risk-free rate $r$ and volatility $\sigma$
- No dividends (we generalize later)
- No arbitrage

### 2.2 Delta Hedging Argument

Let $V(S, t)$ be the price of a derivative. Construct a portfolio:

$$\Pi = V(S, t) - \Delta \cdot S$$

Apply Ito's lemma to $V(S, t)$:

$$dV = \frac{\partial V}{\partial t}\,dt + \frac{\partial V}{\partial S}\,dS + \frac{1}{2}\frac{\partial^2 V}{\partial S^2}(dS)^2$$

Since $(dS)^2 = \sigma^2 S^2 \, dt$ (using the Ito multiplication rules $dW^2 = dt$, $dW \cdot dt = 0$, $dt^2 = 0$):

$$dV = \left(\frac{\partial V}{\partial t} + \mu S \frac{\partial V}{\partial S} + \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2}\right)dt + \sigma S \frac{\partial V}{\partial S}\,dW_t$$

The portfolio change is:

$$d\Pi = dV - \Delta \, dS = \left(\frac{\partial V}{\partial t} + \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} + (\mu S \frac{\partial V}{\partial S} - \Delta \mu S)\right)dt + \sigma S\left(\frac{\partial V}{\partial S} - \Delta\right)dW_t$$

**Choose** $\Delta = \frac{\partial V}{\partial S}$ to eliminate the stochastic term:

$$d\Pi = \left(\frac{\partial V}{\partial t} + \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2}\right)dt$$

This portfolio is instantaneously riskless, so by no-arbitrage it must earn the risk-free rate:

$$d\Pi = r \Pi \, dt = r\left(V - S\frac{\partial V}{\partial S}\right)dt$$

### 2.3 The Black-Scholes PDE

Equating the two expressions for $d\Pi$:

$$\boxed{\frac{\partial V}{\partial t} + \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} + rS\frac{\partial V}{\partial S} - rV = 0}$$

This is the **Black-Scholes PDE**. Observe that $\mu$ has vanished --- the derivative price is independent of the stock's expected return. This is the essence of risk-neutral pricing.

**Boundary conditions for a European call** with strike $K$ and maturity $T$:
- Terminal: $V(S, T) = \max(S - K, 0)$
- As $S \to 0$: $V(0, t) = 0$
- As $S \to \infty$: $V(S, t) \sim S - Ke^{-r(T-t)}$

### 2.4 Solution via Heat Equation Transformation

We solve the BS PDE by transforming it into the heat equation. Define:

$$S = K e^x, \qquad t = T - \frac{2\tau}{\sigma^2}, \qquad V(S,t) = K \, v(x, \tau)$$

**Step 1.** Compute the partial derivatives. Let $\alpha = \frac{2r}{\sigma^2}$:

$$\frac{\partial V}{\partial t} = -\frac{\sigma^2 K}{2}\frac{\partial v}{\partial \tau}$$

$$\frac{\partial V}{\partial S} = \frac{K}{S}\frac{\partial v}{\partial x} = \frac{1}{e^x}\frac{\partial v}{\partial x}$$

$$\frac{\partial^2 V}{\partial S^2} = \frac{K}{S^2}\left(\frac{\partial^2 v}{\partial x^2} - \frac{\partial v}{\partial x}\right)$$

**Step 2.** Substitute into the BS PDE and simplify:

$$\frac{\partial v}{\partial \tau} = \frac{\partial^2 v}{\partial x^2} + (\alpha - 1)\frac{\partial v}{\partial x} - \alpha \, v$$

**Step 3.** Remove the first-order and zeroth-order terms with:

$$v(x, \tau) = e^{\beta x + \gamma \tau} \, u(x, \tau)$$

where $\beta = -\frac{1}{2}(\alpha - 1)$ and $\gamma = -\frac{1}{4}(\alpha + 1)^2$. Substitution yields the **heat equation**:

$$\boxed{\frac{\partial u}{\partial \tau} = \frac{\partial^2 u}{\partial x^2}}$$

**Step 4.** The initial condition for a call becomes:

$$u(x, 0) = e^{-\beta x}\max(e^x - 1, 0) = \max\!\left(e^{\frac{1}{2}(\alpha+1)x} - e^{\frac{1}{2}(\alpha-1)x},\; 0\right)$$

**Step 5.** Apply the fundamental solution of the heat equation (Green's function):

$$u(x, \tau) = \frac{1}{2\sqrt{\pi \tau}} \int_{-\infty}^{\infty} u(y, 0) \, e^{-(x-y)^2/(4\tau)} \, dy$$

**Step 6.** Evaluate the integral by splitting at $y = 0$ and completing the square. After substitution $z = (y - x)/(2\sqrt{\tau})$ and careful algebra, transform back to the original variables to obtain the Black-Scholes formula. The complete evaluation is carried out in Section 3.

---

## 3. Black-Scholes via Risk-Neutral Pricing

### 3.1 The Martingale Approach

By the First Fundamental Theorem of Asset Pricing (Module 08), the absence of arbitrage implies the existence of an equivalent martingale measure $\mathbb{Q}$ under which discounted asset prices are martingales. Under $\mathbb{Q}$:

$$dS_t = rS_t\,dt + \sigma S_t\,dW_t^{\mathbb{Q}}$$

where $W_t^{\mathbb{Q}}$ is a $\mathbb{Q}$-Brownian motion (by Girsanov's theorem, $dW_t^{\mathbb{Q}} = dW_t + \frac{\mu - r}{\sigma}dt$). The solution is:

$$S_T = S_0 \exp\!\left[\left(r - \tfrac{1}{2}\sigma^2\right)T + \sigma W_T^{\mathbb{Q}}\right]$$

The price of any attainable European claim with payoff $\Phi(S_T)$ is:

$$V_0 = e^{-rT}\,\mathbb{E}^{\mathbb{Q}}\!\left[\Phi(S_T)\right]$$

### 3.2 Computing the Call Price

For a European call, $\Phi(S_T) = (S_T - K)^+$:

$$C_0 = e^{-rT}\,\mathbb{E}^{\mathbb{Q}}\!\left[(S_T - K)^+\right]$$

Write $S_T = S_0 e^{(r - \sigma^2/2)T + \sigma\sqrt{T}\,Z}$ where $Z \sim \mathcal{N}(0,1)$ under $\mathbb{Q}$. The call is in the money when $S_T > K$, i.e., when:

$$Z > -\frac{\ln(S_0/K) + (r - \sigma^2/2)T}{\sigma\sqrt{T}} = -d_2$$

where we define:

$$d_1 = \frac{\ln(S_0/K) + (r + \sigma^2/2)T}{\sigma\sqrt{T}}, \qquad d_2 = d_1 - \sigma\sqrt{T}$$

Split the expectation:

$$C_0 = e^{-rT}\!\left[S_0 e^{rT}\,\mathbb{E}^{\mathbb{Q}}\!\left[e^{-\sigma^2 T/2 + \sigma\sqrt{T}Z}\,\mathbf{1}_{Z > -d_2}\right] - K\,\mathbb{P}^{\mathbb{Q}}(Z > -d_2)\right]$$

**Second term:**

$$K\,\mathbb{P}^{\mathbb{Q}}(Z > -d_2) = K\,\Phi(d_2)$$

where $\Phi$ is the standard normal CDF (using symmetry $\mathbb{P}(Z > -a) = \Phi(a)$).

**First term:** Complete the square in the exponent. Let $W = Z - \sigma\sqrt{T}$:

$$\mathbb{E}\!\left[e^{-\sigma^2 T/2 + \sigma\sqrt{T}Z}\,\mathbf{1}_{Z > -d_2}\right] = \mathbb{E}\!\left[e^{\sigma\sqrt{T}W + \sigma^2 T/2 - \sigma^2 T/2}\,\mathbf{1}_{W > -d_2 - \sigma\sqrt{T}}\right]$$

Wait --- more directly, note $-\frac{1}{2}\sigma^2 T + \sigma\sqrt{T}Z = \sigma\sqrt{T}(Z - \sigma\sqrt{T}/1) + \sigma^2 T/2 - \sigma^2 T$. Let us use the moment generating property. We have:

$$\mathbb{E}\!\left[e^{\sigma\sqrt{T}Z - \sigma^2 T/2}\,\mathbf{1}_{Z > -d_2}\right]$$

Under the substitution $\tilde{Z} = Z - \sigma\sqrt{T}$, by Girsanov-style change of measure:

$$= \mathbb{E}^{\tilde{\mathbb{Q}}}\!\left[\mathbf{1}_{\tilde{Z} > -d_2 - \sigma\sqrt{T}}\right] = \mathbb{P}^{\tilde{\mathbb{Q}}}(\tilde{Z} > -d_1) = \Phi(d_1)$$

where $\tilde{Z} \sim \mathcal{N}(0,1)$ under $\tilde{\mathbb{Q}}$ and $d_1 = d_2 + \sigma\sqrt{T}$.

Combining:

$$\boxed{C_0 = S_0\,\Phi(d_1) - K e^{-rT}\,\Phi(d_2)}$$

---

## 4. Black-Scholes Closed-Form Formulae

### 4.1 European Call and Put

With $\tau = T - t$ denoting time to expiry:

$$d_1 = \frac{\ln(S/K) + (r + \sigma^2/2)\tau}{\sigma\sqrt{\tau}}, \qquad d_2 = d_1 - \sigma\sqrt{\tau}$$

| Option | Price |
|--------|-------|
| **European Call** | $C = S\,\Phi(d_1) - Ke^{-r\tau}\Phi(d_2)$ |
| **European Put** | $P = Ke^{-r\tau}\Phi(-d_2) - S\,\Phi(-d_1)$ |

### 4.2 With Continuous Dividend Yield $\delta$

Replace $S$ with $Se^{-\delta\tau}$ in the formulae (Merton, 1973):

$$C = Se^{-\delta\tau}\Phi(d_1) - Ke^{-r\tau}\Phi(d_2)$$

where now:

$$d_1 = \frac{\ln(S/K) + (r - \delta + \sigma^2/2)\tau}{\sigma\sqrt{\tau}}$$

### 4.3 Digital (Binary) Options

A **cash-or-nothing call** pays 1 if $S_T > K$:

$$V_{\text{CoN,call}} = e^{-r\tau}\,\Phi(d_2)$$

An **asset-or-nothing call** pays $S_T$ if $S_T > K$:

$$V_{\text{AoN,call}} = S\,\Phi(d_1)$$

Note: $C = V_{\text{AoN,call}} - K \cdot V_{\text{CoN,call}}$, providing a natural decomposition.

The corresponding put variants:

$$V_{\text{CoN,put}} = e^{-r\tau}\,\Phi(-d_2), \qquad V_{\text{AoN,put}} = S\,\Phi(-d_1)$$

### 4.4 Summary of Standard Notation

Throughout this module we use:
- $\phi(x) = \frac{1}{\sqrt{2\pi}}e^{-x^2/2}$ --- standard normal PDF
- $\Phi(x) = \int_{-\infty}^{x}\phi(z)\,dz$ --- standard normal CDF
- Key identity: $S\,\phi(d_1) = Ke^{-r\tau}\phi(d_2)$ (used repeatedly in Greek derivations)

**Proof of the identity:** $\phi(d_1)/\phi(d_2) = \exp\!\left(-\frac{d_1^2 - d_2^2}{2}\right)$. Since $d_1^2 - d_2^2 = (d_1 - d_2)(d_1 + d_2) = \sigma\sqrt{\tau}\cdot 2d_1 - \sigma^2\tau$, expanding yields $\frac{d_1^2 - d_2^2}{2} = \ln(S/K) + r\tau$, so $\phi(d_1) = \frac{Ke^{-r\tau}}{S}\phi(d_2)$. $\blacksquare$

---

## 5. The Greeks: Complete Taxonomy

The Greeks measure the sensitivity of option prices to model parameters. We derive each from the Black-Scholes call formula $C = S\Phi(d_1) - Ke^{-r\tau}\Phi(d_2)$.

### 5.1 First-Order Greeks

**Delta** ($\Delta = \frac{\partial C}{\partial S}$):

$$\Delta_{\text{call}} = \Phi(d_1)$$

*Derivation:* $\frac{\partial C}{\partial S} = \Phi(d_1) + S\phi(d_1)\frac{\partial d_1}{\partial S} - Ke^{-r\tau}\phi(d_2)\frac{\partial d_2}{\partial S}$. Since $\frac{\partial d_1}{\partial S} = \frac{\partial d_2}{\partial S} = \frac{1}{S\sigma\sqrt{\tau}}$ and using the identity $S\phi(d_1) = Ke^{-r\tau}\phi(d_2)$, the last two terms cancel.

$$\Delta_{\text{put}} = \Phi(d_1) - 1 = -\Phi(-d_1)$$

**Vega** ($\mathcal{V} = \frac{\partial C}{\partial \sigma}$):

$$\mathcal{V} = S\sqrt{\tau}\,\phi(d_1) = Ke^{-r\tau}\sqrt{\tau}\,\phi(d_2) > 0$$

*Derivation:* $\frac{\partial C}{\partial \sigma} = S\phi(d_1)\frac{\partial d_1}{\partial \sigma} - Ke^{-r\tau}\phi(d_2)\frac{\partial d_2}{\partial \sigma}$. We have $\frac{\partial d_1}{\partial \sigma} = -\frac{d_2}{\sigma}$ and $\frac{\partial d_2}{\partial \sigma} = -\frac{d_1}{\sigma}$. Using the identity and simplifying: $\mathcal{V} = S\phi(d_1)\left(-\frac{d_2}{\sigma} + \frac{d_1}{\sigma}\right) = S\phi(d_1)\sqrt{\tau}$.

Vega is identical for calls and puts (by put-call parity, since $\frac{\partial}{\partial \sigma}(S - Ke^{-r\tau}) = 0$).

**Theta** ($\Theta = \frac{\partial C}{\partial t} = -\frac{\partial C}{\partial \tau}$):

$$\Theta_{\text{call}} = -\frac{S\sigma\phi(d_1)}{2\sqrt{\tau}} - rKe^{-r\tau}\Phi(d_2)$$

$$\Theta_{\text{put}} = -\frac{S\sigma\phi(d_1)}{2\sqrt{\tau}} + rKe^{-r\tau}\Phi(-d_2)$$

**Rho** ($\rho = \frac{\partial C}{\partial r}$):

$$\rho_{\text{call}} = K\tau e^{-r\tau}\Phi(d_2)$$

$$\rho_{\text{put}} = -K\tau e^{-r\tau}\Phi(-d_2)$$

### 5.2 Second-Order Greeks

**Gamma** ($\Gamma = \frac{\partial^2 C}{\partial S^2} = \frac{\partial \Delta}{\partial S}$):

$$\Gamma = \frac{\phi(d_1)}{S\sigma\sqrt{\tau}} > 0$$

Gamma is identical for calls and puts. It peaks for ATM options near expiry.

**Vanna** ($\frac{\partial \Delta}{\partial \sigma} = \frac{\partial \mathcal{V}}{\partial S}$):

$$\text{Vanna} = -\frac{\phi(d_1) \, d_2}{\sigma} = \frac{\mathcal{V}}{S}\left(1 - \frac{d_1}{\sigma\sqrt{\tau}}\right)$$

*Derivation:* $\text{Vanna} = \frac{\partial}{\partial \sigma}\Phi(d_1) = \phi(d_1)\frac{\partial d_1}{\partial \sigma} = \phi(d_1)\cdot\left(-\frac{d_2}{\sigma}\right)$.

Vanna measures how delta changes with volatility --- critical for volatility hedging.

**Volga (Vomma)** ($\frac{\partial^2 C}{\partial \sigma^2} = \frac{\partial \mathcal{V}}{\partial \sigma}$):

$$\text{Volga} = \mathcal{V}\,\frac{d_1 d_2}{\sigma} = S\sqrt{\tau}\,\phi(d_1)\,\frac{d_1 d_2}{\sigma}$$

Volga is the vol-of-vol sensitivity --- it drives the convexity of option prices in volatility.

**Charm (Delta Bleed)** ($\frac{\partial \Delta}{\partial t} = -\frac{\partial \Delta}{\partial \tau}$):

$$\text{Charm}_{\text{call}} = -\phi(d_1)\left(\frac{2r\tau - d_2\sigma\sqrt{\tau}}{2\tau\sigma\sqrt{\tau}}\right)$$

This measures how delta drifts as time passes --- essential for understanding hedging costs overnight.

**Veta** ($\frac{\partial \mathcal{V}}{\partial t} = -\frac{\partial \mathcal{V}}{\partial \tau}$):

$$\text{Veta} = -S\phi(d_1)\sqrt{\tau}\left[\frac{d_1\left(r - \frac{d_1\sigma}{2\sqrt{\tau}}\right)}{\sigma\sqrt{\tau}} + \frac{1 - d_1 d_2}{2\tau}\right]$$

A simplified form: $\text{Veta} = \mathcal{V}\left[\frac{r\,d_1}{\sigma\sqrt{\tau}} - \frac{1 + d_1 d_2}{2\tau}\right]$

### 5.3 Third-Order Greeks

**Speed** ($\frac{\partial \Gamma}{\partial S} = \frac{\partial^3 C}{\partial S^3}$):

$$\text{Speed} = -\frac{\Gamma}{S}\left(\frac{d_1}{\sigma\sqrt{\tau}} + 1\right)$$

Speed measures the rate of change of gamma with respect to the underlying --- relevant for gamma scalping.

**Color (Gamma Bleed)** ($\frac{\partial \Gamma}{\partial t} = -\frac{\partial \Gamma}{\partial \tau}$):

$$\text{Color} = -\frac{\phi(d_1)}{2S\tau\sigma\sqrt{\tau}}\left[1 + d_1\cdot\frac{2r\tau - d_2\sigma\sqrt{\tau}}{\sigma\sqrt{\tau}}\right]$$

**Zomma** ($\frac{\partial \Gamma}{\partial \sigma}$):

$$\text{Zomma} = \Gamma \cdot \frac{d_1 d_2 - 1}{\sigma}$$

Zomma is important when gamma-hedging in a stochastic volatility environment.

**Ultima** ($\frac{\partial^3 C}{\partial \sigma^3}$):

$$\text{Ultima} = -\frac{\mathcal{V}}{\sigma^2}\left[d_1 d_2(1 - d_1 d_2) + d_1^2 + d_2^2\right]$$

### 5.4 Complete Greeks Reference Table

| Greek | Symbol | Formula (European Call) | Order |
|-------|--------|------------------------|-------|
| **Delta** | $\partial C/\partial S$ | $\Phi(d_1)$ | 1st |
| **Vega** | $\partial C/\partial \sigma$ | $S\sqrt{\tau}\,\phi(d_1)$ | 1st |
| **Theta** | $\partial C/\partial t$ | $-\frac{S\sigma\phi(d_1)}{2\sqrt{\tau}} - rKe^{-r\tau}\Phi(d_2)$ | 1st |
| **Rho** | $\partial C/\partial r$ | $K\tau e^{-r\tau}\Phi(d_2)$ | 1st |
| **Gamma** | $\partial^2 C/\partial S^2$ | $\frac{\phi(d_1)}{S\sigma\sqrt{\tau}}$ | 2nd |
| **Vanna** | $\partial^2 C/\partial S\partial\sigma$ | $-\frac{\phi(d_1)d_2}{\sigma}$ | 2nd |
| **Volga** | $\partial^2 C/\partial\sigma^2$ | $S\sqrt{\tau}\,\phi(d_1)\frac{d_1 d_2}{\sigma}$ | 2nd |
| **Charm** | $\partial^2 C/\partial S\partial t$ | $-\phi(d_1)\frac{2r\tau - d_2\sigma\sqrt{\tau}}{2\tau\sigma\sqrt{\tau}}$ | 2nd |
| **Veta** | $\partial^2 C/\partial\sigma\partial t$ | $\mathcal{V}\left[\frac{r d_1}{\sigma\sqrt{\tau}} - \frac{1+d_1 d_2}{2\tau}\right]$ | 2nd |
| **Speed** | $\partial^3 C/\partial S^3$ | $-\frac{\Gamma}{S}\left(\frac{d_1}{\sigma\sqrt{\tau}}+1\right)$ | 3rd |
| **Color** | $\partial^3 C/\partial S^2\partial t$ | $-\frac{\phi(d_1)}{2S\tau\sigma\sqrt{\tau}}\!\left[1+d_1\frac{2r\tau-d_2\sigma\sqrt{\tau}}{\sigma\sqrt{\tau}}\right]$ | 3rd |
| **Zomma** | $\partial^3 C/\partial S^2\partial\sigma$ | $\Gamma\frac{d_1 d_2-1}{\sigma}$ | 3rd |
| **Ultima** | $\partial^3 C/\partial\sigma^3$ | $-\frac{\mathcal{V}}{\sigma^2}[d_1 d_2(1-d_1 d_2)+d_1^2+d_2^2]$ | 3rd |

```mermaid
graph LR
    C["Option Price C"]
    C -->|"∂/∂S"| Delta["Delta Δ"]
    C -->|"∂/∂σ"| Vega["Vega ν"]
    C -->|"∂/∂t"| Theta["Theta Θ"]
    C -->|"∂/∂r"| Rho["Rho ρ"]

    Delta -->|"∂/∂S"| Gamma["Gamma Γ"]
    Delta -->|"∂/∂σ"| Vanna["Vanna"]
    Delta -->|"∂/∂t"| Charm["Charm"]

    Vega -->|"∂/∂σ"| Volga["Volga"]
    Vega -->|"∂/∂t"| Veta["Veta"]

    Gamma -->|"∂/∂S"| Speed["Speed"]
    Gamma -->|"∂/∂t"| Color["Color"]
    Gamma -->|"∂/∂σ"| Zomma["Zomma"]

    Volga -->|"∂/∂σ"| Ultima["Ultima"]

    style C fill:#1a73e8,color:#fff
    style Delta fill:#34a853,color:#fff
    style Vega fill:#34a853,color:#fff
    style Theta fill:#34a853,color:#fff
    style Rho fill:#34a853,color:#fff
    style Gamma fill:#fbbc04,color:#000
    style Vanna fill:#fbbc04,color:#000
    style Volga fill:#fbbc04,color:#000
    style Charm fill:#fbbc04,color:#000
    style Veta fill:#fbbc04,color:#000
    style Speed fill:#ea4335,color:#fff
    style Color fill:#ea4335,color:#fff
    style Zomma fill:#ea4335,color:#fff
    style Ultima fill:#ea4335,color:#fff
```

### 5.5 The Black-Scholes PDE Rewritten in Greeks

The BS PDE takes a remarkably clean form:

$$\Theta + \frac{1}{2}\sigma^2 S^2 \Gamma + rS\Delta - rV = 0$$

This encodes the fundamental trade-off: theta (time decay) pays for gamma (convexity). A delta-hedged long option position earns gamma P&L $\frac{1}{2}\Gamma(\Delta S)^2$ and pays theta.

---

## 6. Implied Volatility

### 6.1 Definition

The **implied volatility** $\sigma_{\text{imp}}$ of a European option with observed market price $C_{\text{mkt}}$ is the unique $\sigma > 0$ satisfying:

$$C_{\text{BS}}(S, K, \tau, r, \sigma_{\text{imp}}) = C_{\text{mkt}}$$

Uniqueness is guaranteed because vega $> 0$: the BS price is strictly increasing in $\sigma$.

### 6.2 Newton-Raphson Inversion

The standard numerical approach exploits the known vega:

$$\sigma_{n+1} = \sigma_n - \frac{C_{\text{BS}}(\sigma_n) - C_{\text{mkt}}}{\mathcal{V}(\sigma_n)}$$

This converges quadratically near the root. **Practical notes:**
- Initial guess: use the Brenner-Subrahmanyam approximation (below)
- Guard against $\mathcal{V} \approx 0$ near expiry or deep ITM/OTM
- Typical convergence: 4--6 iterations to machine precision

### 6.3 Brenner-Subrahmanyam Approximation

For an ATM option ($S \approx Ke^{-r\tau}$):

$$\sigma_{\text{imp}} \approx \frac{C_{\text{mkt}}}{S}\sqrt{\frac{2\pi}{\tau}}$$

This follows from the ATM approximation $C_{\text{ATM}} \approx S\sigma\sqrt{\tau/(2\pi)}$ (first-order Taylor expansion of $\Phi$ around 0).

### 6.4 Jaeckel's Rational Approximation

Peter Jaeckel (2015) developed a method achieving machine-precision implied volatility in at most two iterations by using:

1. A normalized option price $\beta = C/(S\sqrt{\tau})$ (intrinsic value removed)
2. Rational function approximations to the inverse of the normalized BS formula in different regimes (near-ATM, OTM, deep OTM)
3. Householder's method of order 4 for refinement

The key insight is transforming to coordinates where the BS formula is nearly linear, making rational approximation extremely accurate. The algorithm handles all moneyness regimes without branching failures and is the industry standard for implied vol computation.

```mermaid
flowchart TD
    A["Market Price C_mkt"] --> B{"Moneyness<br/>regime?"}
    B -->|"Near ATM"| C["Brenner-Subrahmanyam<br/>initial guess"]
    B -->|"OTM/ITM"| D["Rational approximation<br/>initial guess"]
    B -->|"Deep OTM"| E["Asymptotic expansion<br/>initial guess"]
    C --> F["Newton-Raphson<br/>iteration"]
    D --> F
    E --> F
    F --> G{"|f| < tol?"}
    G -->|"No"| F
    G -->|"Yes"| H["σ_imp"]

    style A fill:#1a73e8,color:#fff
    style H fill:#34a853,color:#fff
```

---

## 7. Volatility Smile and Skew

### 7.1 Empirical Observations

If the Black-Scholes model were correct, implied volatility would be constant across strikes and maturities. The empirical record contradicts this decisively.

- **Equity indices** exhibit a pronounced **volatility skew** (often called the "smirk"): $\sigma_{\text{imp}}$ decreases monotonically with strike. OTM puts trade at substantially higher implied vols than OTM calls. For the S&P 500, the 25-delta put implied vol can exceed the 25-delta call by 5--10 vol points. This pattern became dramatically steeper after the 1987 crash, when market participants collectively repriced downside tail risk. The economic explanation combines the **leverage effect** (as $S$ falls, the firm's equity-to-debt ratio drops, increasing effective volatility) with **crash-risk aversion** (investors pay a premium for portfolio insurance in the form of OTM puts).

- **FX markets** display a **volatility smile**: both OTM puts and OTM calls have higher implied vol than ATM options. This symmetric pattern reflects the fact that neither currency in a pair has a natural "downside" --- large moves in either direction are equally plausible. The smile is typically parameterized in terms of delta rather than strike, with ATM defined by the delta-neutral straddle convention.

- **Commodity markets** show skews that depend on the specific supply/demand dynamics. Energy markets often exhibit right-skew (calls more expensive) reflecting supply disruption risk, while agricultural commodities may show left-skew due to bumper crop fears.

- **Interest rate markets** display complex term-structure-dependent smile behavior. Short-dated swaptions tend to show pronounced skew, while long-dated options exhibit more symmetric smiles.

The smile is not merely a static cross-sectional phenomenon. It also has a **term structure**: short-dated options tend to have steeper smiles than long-dated options, because the CLT smooths the return distribution over longer horizons, reducing effective kurtosis.

### 7.2 What Black-Scholes Gets Wrong

The BS model assumes log-normal returns with constant volatility. Empirical return distributions differ from this assumption in several systematic ways:

1. **Fat tails (leptokurtosis):** Return distributions have excess kurtosis $\gg 0$. Daily equity index returns typically have kurtosis around 5--10 (vs. 3 for a normal distribution). This means extreme events --- moves of 5 or more standard deviations --- occur orders of magnitude more frequently than predicted by the Gaussian model. The October 1987 crash was roughly a 20-sigma event under log-normal assumptions.

2. **Negative skewness:** Large downward moves are more frequent and severe than large upward moves, at least for equity indices. The distribution of daily returns exhibits skewness around $-0.5$ to $-1.0$, indicating an asymmetric left tail.

3. **Stochastic volatility:** Volatility is not constant. It clusters in time (high-vol regimes persist), is mean-reverting, and is typically negatively correlated with returns for equities (the "leverage effect"). Realized volatility can vary by a factor of 5--10 between calm and crisis periods.

4. **Jumps:** Price processes exhibit discontinuous jumps, especially during earnings announcements, macroeconomic data releases, and geopolitical events. These jumps produce excess kurtosis in short-dated returns that cannot be explained by continuous stochastic volatility alone.

5. **Volatility of volatility:** Even the dynamics of volatility exhibit randomness. The "vol-of-vol" drives the convexity (curvature) of the smile and is captured by second-order vol Greeks like volga.

Each of these phenomena leaves a distinct fingerprint on the implied volatility surface. The smile's slope (skew) relates primarily to the correlation between returns and volatility, while its curvature (convexity) relates to the kurtosis and vol-of-vol.

### 7.3 Quantifying the Smile: Risk-Reversal and Butterfly

Rather than quoting the full implied volatility surface, practitioners compress it into a small number of key metrics at each maturity.

**ATM Volatility:** The level of the smile, typically defined at the delta-neutral straddle strike (the strike where the straddle has zero delta):

$$\sigma_{\text{ATM}} = \sigma_{\text{imp}}(K_{\text{ATM}})$$

This anchors the overall volatility level and is the most liquid point on the surface.

**Risk-Reversal (RR):** Measures the skew --- the slope of the smile:

$$\text{RR}_{25} = \sigma_{\text{imp}}(25\Delta\text{-call}) - \sigma_{\text{imp}}(25\Delta\text{-put})$$

A negative RR for equities reflects the skew (OTM puts are relatively expensive). The RR is directly tradeable: buying a risk-reversal means buying the OTM call and selling the OTM put at the same delta. It provides leveraged exposure to the direction of the skew.

**Butterfly (BF):** Measures the curvature --- the convexity of the smile:

$$\text{BF}_{25} = \frac{\sigma_{\text{imp}}(25\Delta\text{-call}) + \sigma_{\text{imp}}(25\Delta\text{-put})}{2} - \sigma_{\text{imp}}(\text{ATM})$$

A positive BF means the wings are elevated relative to ATM, indicating fat tails (excess kurtosis). The butterfly spread is a direct bet on the curvature of the smile: buying the 25-delta strangle and selling the ATM straddle, with notionals adjusted to be vega-neutral.

**Connection to moments:** To first order, the smile parameters relate to the risk-neutral distribution:
- ATM vol $\approx$ standard deviation of log-returns
- RR $\propto$ skewness of the risk-neutral distribution
- BF $\propto$ excess kurtosis of the risk-neutral distribution

This connection is made precise by the Backus-Foresi-Wu (2004) expansion.

### 7.4 Smile Dynamics

The behavior of the smile over time is described by two competing heuristics:

- **Sticky strike:** $\sigma_{\text{imp}}(K)$ stays fixed as $S$ moves. The smile is anchored to absolute strike levels. Under this assumption, if $S$ drops by 5%, the ATM vol increases (because the ATM strike is now closer to where the high-vol puts used to be). Sticky strike implies local vol dynamics (see Module 19).

- **Sticky delta:** $\sigma_{\text{imp}}(\Delta)$ stays fixed as $S$ moves. The smile translates with the underlying. Under this assumption, a 5% drop in $S$ leaves ATM vol unchanged, because the smile has shifted down with $S$. Sticky delta is more consistent with stochastic volatility models.

- **Sticky local volatility:** $\sigma_{\text{loc}}(S, t)$ stays fixed. This is the prediction of Dupire's local volatility model and implies that sticky strike holds for the smile, but with specific dynamics for how skew evolves over time.

In practice, neither heuristic is perfectly correct. Empirical evidence suggests that equity index smiles behave somewhere between sticky strike and sticky delta, with the smile "sliding" partially along with the underlying. The true dynamics depend on the underlying stochastic volatility process (see Module 19 for local vol, SABR, Heston).

Understanding smile dynamics is critical for hedging. A trader who delta-hedges using a sticky-strike assumption will take different positions than one using sticky-delta, leading to different P&L profiles when the underlying moves.

---

## 8. Exotic Options

### 8.1 Barrier Options

A **barrier option** is activated or extinguished when the underlying hits a specified level $H$.

**Types:**
- **Knock-in:** Comes alive when barrier is hit
- **Knock-out:** Dies when barrier is hit
- **Up / Down:** Barrier above / below current spot

**In-Out Parity:** For the same strike and barrier, a knock-in and knock-out pair must satisfy:

$$V_{\text{KI}} + V_{\text{KO}} = V_{\text{vanilla}}$$

This follows because exactly one of the two barriers is activated on every path: either the barrier is hit (knock-in lives, knock-out dies) or it is not (knock-in dies, knock-out lives). Therefore, holding both always produces the vanilla payoff.

**Up-and-Out Call ($H > K$) via Reflection Principle:**

For an up-and-out call with barrier $H > S_0$, define $\lambda = \frac{r - \sigma^2/2}{\sigma^2}$. The price is:

$$C_{\text{UO}} = C_{\text{BS}}(S, K) - \left(\frac{H}{S}\right)^{2\lambda}\!\! C_{\text{BS}}\!\left(\frac{H^2}{S}, K\right) \qquad \text{(for } K < H\text{)}$$

The derivation uses the **reflection principle** for Brownian motion. Consider the log-price process $X_t = \ln S_t$. Under the risk-neutral measure, $X_t$ is a Brownian motion with drift $\mu_Q = r - \sigma^2/2$.

The key observation is this: for a driftless Brownian motion $B_t$, the probability that a path starting at $x < h$ hits the barrier $h$ and ends at some level $y < h$ is related by the reflection principle to the probability that a path starting at the reflected point $2h - x$ (which is above $h$) ends at $y$ without any barrier. This is because reflecting a path that hits $h$ about the line $y = h$ creates a bijection between paths from $x$ that touch $h$ and paths from $2h - x$ that do not necessarily touch $h$.

For a Brownian motion with drift, the reflection symmetry is broken, but it can be restored by a Girsanov change of measure. The Radon-Nikodym derivative introduces the factor $(H/S)^{2\lambda}$, which adjusts for the drift. The reflected starting point $H^2/S$ in price space corresponds to $2\ln H - \ln S$ in log-price space.

**Practical consideration:** The closed-form formula assumes continuous barrier monitoring. Discretely monitored barriers (checked daily, for instance) lead to systematically different prices. Broadie, Glasserman, and Kou (1997) provide a continuity correction: replace $H$ with $He^{\pm 0.5826 \sigma\sqrt{\delta t}}$ (sign depends on barrier direction), where $\delta t$ is the monitoring interval.

### 8.2 Asian Options

An **Asian option** has payoff depending on the average price $\bar{S} = \frac{1}{n}\sum_{i=1}^n S_{t_i}$ (discrete) or $\bar{S} = \frac{1}{T}\int_0^T S_t\,dt$ (continuous). Asian options are popular in commodity and FX markets because the averaging reduces manipulation risk and smooths the payoff against short-term price spikes.

**Geometric Asian:** The geometric average $\tilde{S} = \left(\prod_{i=1}^n S_{t_i}\right)^{1/n}$ is log-normal (since the log of a product is a sum of log-normals, and a weighted sum of jointly normal variables is normal), permitting a Black-Scholes-type closed-form solution:

$$C_{\text{geo}} = e^{-rT}\mathbb{E}^{\mathbb{Q}}\!\left[(\tilde{S} - K)^+\right]$$

with $\ln\tilde{S} \sim \mathcal{N}(\tilde{\mu}, \tilde{\sigma}^2)$ where:

$$\tilde{\mu} = \frac{1}{n}\sum_{i=1}^{n}\left[\ln S_0 + \left(r - \frac{\sigma^2}{2}\right)t_i\right]$$

$$\tilde{\sigma}^2 = \frac{\sigma^2}{n^2}\sum_{i=1}^{n}\sum_{j=1}^{n}\min(t_i, t_j)$$

The price then follows the standard BS formula with adjusted drift and volatility.

**Arithmetic Asian:** No closed form exists because the sum of log-normals is not log-normal. This is a fundamental analytical obstruction. Standard approaches include:

- **Monte Carlo simulation** with control variates using the geometric Asian. Since the geometric Asian has a known closed-form price, we can reduce variance dramatically by computing $\hat{C}_{\text{arith}} = \hat{C}_{\text{arith,MC}} - \beta(\hat{C}_{\text{geo,MC}} - C_{\text{geo,exact}})$ where $\beta$ is the regression coefficient. Variance reduction factors of 50--100x are typical.
- **Moment matching** (Turnbull and Wakeman, 1991): approximate the arithmetic average as log-normal by matching the first two moments. This yields a simple but approximate formula.
- **PDE methods:** The state space grows by one dimension for the running average. Define $A_t = \int_0^t S_u\,du$ and solve the 2D PDE in $(S, A, t)$. Efficient for single-asset Asians but expensive for baskets.

### 8.3 Lookback Options

A **floating-strike lookback call** has payoff $S_T - \min_{0 \le t \le T} S_t$, giving the holder the benefit of buying at the lowest price over the option's life. This is among the most expensive vanilla-like contracts.

A **fixed-strike lookback call** has payoff $\max(\max_{0 \le t \le T} S_t - K, 0)$, which values the right to sell at the maximum observed price.

For the floating-strike lookback call, using the joint distribution of $(S_T, \min_{0\le t\le T} S_t)$ under the risk-neutral measure (which follows from the reflection principle for drifted Brownian motion):

$$C_{\text{lookback}} = S\,\Phi(a_1) - Se^{-r\tau}\frac{\sigma^2}{2r}\Phi(-a_1) - S_{\min}e^{-r\tau}\!\left[\Phi(a_2) - \frac{\sigma^2}{2r}e^{a_3}\Phi(-a_4)\right]$$

where:

$$a_1 = \frac{\ln(S/S_{\min}) + (r + \sigma^2/2)\tau}{\sigma\sqrt{\tau}}, \qquad a_2 = a_1 - \sigma\sqrt{\tau}$$

$$a_3 = \frac{2(r - \sigma^2/2)\ln(S/S_{\min})}{\sigma^2}, \qquad a_4 = a_1 - \frac{2r\sqrt{\tau}}{\sigma}$$

(Goldman, Sosin, Gatto 1979). Like barrier options, lookbacks are sensitive to monitoring frequency. Continuously monitored lookbacks are much more expensive than their discretely monitored counterparts, because the continuous maximum/minimum is almost surely more extreme.

### 8.4 Basket Options

A **basket option** has payoff based on a weighted sum $B_T = \sum_i w_i S_T^{(i)}$ of $d$ underlying assets. These are extensively used for index options and multi-asset structured products.

No closed form exists in general (the same analytical obstruction as arithmetic Asians --- the sum of log-normals is not log-normal). The difficulty grows with the number of assets and the complexity of the correlation structure. Approaches include:

- **Monte Carlo with correlated samples:** Generate correlated Brownian increments via Cholesky decomposition of the $d \times d$ correlation matrix $\Sigma = LL^T$. If $Z_i$ are independent standard normals, then $W = LZ$ has the correct correlation structure. For large $d$, this is the method of choice.
- **Moment matching** (Levy, 1992): Approximate $B_T$ as log-normal by matching the first two moments: $\mathbb{E}[B_T] = \sum_i w_i \mathbb{E}[S_T^{(i)}]$ and $\text{Var}(B_T) = \sum_{i,j} w_i w_j \text{Cov}(S_T^{(i)}, S_T^{(j)})$. This gives an approximate BS-like formula.
- **Copula-based PDE methods:** For 2--3 assets, finite difference methods on a structured grid remain tractable. Beyond 3 dimensions, the curse of dimensionality makes PDE methods prohibitive without sparse grids or other dimension reduction techniques.

### 8.5 Pricing Exotics: MC vs PDE

```mermaid
flowchart LR
    subgraph MC["Monte Carlo"]
        M1["Flexible payoffs"]
        M2["High dimensions OK"]
        M3["Slow convergence O(1/√N)"]
        M4["Path-dependent: natural"]
    end

    subgraph PDE["PDE / FDM"]
        P1["Fast for low dimensions"]
        P2["Curse of dimensionality"]
        P3["Greeks from grid"]
        P4["Early exercise: natural"]
    end

    MC --- Comparison
    PDE --- Comparison

    Comparison["Choose method based on<br/>dimension, payoff, need for Greeks"]

    style MC fill:#e8f0fe,color:#000
    style PDE fill:#fce8e6,color:#000
    style Comparison fill:#fff,stroke:#333
```

---

## 9. American Options and Early Exercise

### 9.1 The Early Exercise Problem

An American option can be exercised at any time $t \in [0, T]$. Its price is:

$$V_0 = \sup_{\tau \in \mathcal{T}} \, e^{-r\tau}\,\mathbb{E}^{\mathbb{Q}}\!\left[\Phi(S_\tau)\right]$$

where $\mathcal{T}$ is the set of stopping times adapted to the filtration.

**Key results on early exercise:**

**Result 1: American calls on non-dividend stocks are never exercised early.** For a European call, $C \ge S - Ke^{-r\tau}$ (this follows from put-call parity and the non-negativity of puts). Since $e^{-r\tau} < 1$ when $r > 0$ and $\tau > 0$:

$$C \ge S - Ke^{-r\tau} > S - K$$

The option is worth strictly more alive than dead (its market value exceeds its exercise value). Therefore $V_{\text{Am,call}} = V_{\text{Eu,call}}$ when there are no dividends. This is a model-free result --- it holds for any dynamics, not just Black-Scholes.

**Result 2: American puts may be exercised early.** Consider an extreme case: $S = 0$ (the stock is worthless). The put's exercise value is $K$, but its European value is only $Ke^{-r\tau} < K$. The holder would prefer to exercise immediately and invest the proceeds at the risk-free rate. More generally, for sufficiently deep ITM puts, the forgone interest on the exercise proceeds (which equals $rK\,dt$ per instant) exceeds the option's theta (time value decay), making early exercise optimal.

**Result 3: American calls on dividend-paying stocks may be exercised early.** The discrete dividend reduces the stock price by the dividend amount on the ex-date. Just before the ex-date, the holder must decide: exercise and capture the dividend, or hold and suffer the price drop. Early exercise may be optimal when the dividend exceeds the remaining time value plus the interest earned on the deferred strike payment. In practice, this means checking the early exercise condition at each ex-dividend date.

### 9.2 The Free Boundary Problem

The American put price satisfies the Black-Scholes PDE in the **continuation region** $\{(S, t) : S > S^*(t)\}$, where $S^*(t)$ is the early exercise boundary (also called the critical stock price or optimal exercise frontier). In the **exercise region** $\{(S, t) : S \le S^*(t)\}$, the option value equals the intrinsic value $(K - S)$.

The early exercise boundary $S^*(t)$ is a free boundary --- it is not known in advance but must be determined as part of the solution. It is characterized by two conditions at the boundary:

$$V(S^*(t), t) = K - S^*(t) \qquad \text{(value matching)}$$

$$\frac{\partial V}{\partial S}(S^*(t), t) = -1 \qquad \text{(smooth pasting)}$$

The value matching condition simply states continuity across the boundary. The smooth pasting condition (also called the high-contact condition) states that the derivative is also continuous --- without it, an arbitrage opportunity would exist by exercising slightly earlier or later than optimal.

**Properties of $S^*(t)$:** The boundary is a decreasing function of time to maturity. At expiry, $S^*(T) = K$ (exercise any ITM option). As $\tau \to \infty$, $S^*(t) \to S^*_{\infty} = \frac{K}{1 + \sigma^2/(2r)}$ for the perpetual American put. The boundary is smooth (continuously differentiable) but has no known closed-form expression, necessitating numerical methods.

The early exercise premium can be expressed as an integral involving the boundary (Kim, 1990; Carr, Jarrow, Myneni, 1992):

$$V_{\text{Am}}(S, t) = V_{\text{Eu}}(S, t) + \int_t^T rKe^{-r(u-t)}\,\Phi\!\left(-d_2(S, S^*(u), u-t)\right)\,du$$

This is known as the **early exercise premium representation** and shows that the American price equals the European price plus an integral of the present value of interest earned from early exercise over all future dates, weighted by the probability of being in the exercise region.

### 9.3 Longstaff-Schwartz Least-Squares Monte Carlo (LSM)

The LSM algorithm (Longstaff and Schwartz, 2001) solves the American option problem by backward induction along simulated paths.

**Algorithm:**

1. Simulate $M$ paths of the underlying: $\{S_{t_j}^{(m)}\}$ for $j = 0, 1, \ldots, N$ and $m = 1, \ldots, M$

2. At terminal time $t_N$: set cash flows $F^{(m)} = \Phi(S_{t_N}^{(m)})$

3. At each exercise date $t_j$ (backward from $j = N-1$ to $j = 1$):

   a. Identify in-the-money paths: $\mathcal{I}_j = \{m : \Phi(S_{t_j}^{(m)}) > 0\}$

   b. For in-the-money paths, regress discounted future cash flows on basis functions of $S_{t_j}^{(m)}$:

   $$e^{-r(t_k - t_j)} F^{(m)} = \sum_{p=0}^{P} \alpha_p \, L_p(S_{t_j}^{(m)}) + \epsilon^{(m)}, \qquad m \in \mathcal{I}_j$$

   where $L_p$ are basis functions (Laguerre polynomials, monomials, etc.) and $t_k$ is the time of the cash flow $F^{(m)}$.

   c. The fitted values $\hat{E}_j^{(m)} = \sum_p \hat{\alpha}_p L_p(S_{t_j}^{(m)})$ estimate the **continuation value**

   d. Exercise at $t_j$ if $\Phi(S_{t_j}^{(m)}) > \hat{E}_j^{(m)}$; update $F^{(m)}$ accordingly

4. The option price is: $V_0 = \frac{1}{M}\sum_{m=1}^{M} e^{-r t^*_m} F^{(m)}$ where $t^*_m$ is the optimal exercise time for path $m$

**Justification:** The conditional expectation $\mathbb{E}[e^{-r\Delta t} V_{t_{j+1}} | S_{t_j}]$ lies in $L^2(\sigma(S_{t_j}))$ and can be approximated arbitrarily well by projecting onto a basis of $L^2$. The regression provides this projection.

```mermaid
flowchart TD
    A["Simulate M paths<br/>forward in time"] --> B["Set terminal payoffs"]
    B --> C["Step backward: j = N-1"]
    C --> D["Identify ITM paths"]
    D --> E["Regress discounted future<br/>cash flows on basis functions"]
    E --> F["Compare exercise value<br/>vs continuation value"]
    F --> G["Update cash flows<br/>for exercised paths"]
    G --> H{"j = 1?"}
    H -->|"No"| I["j = j - 1"]
    I --> D
    H -->|"Yes"| J["Discount all cash flows<br/>to time 0, average"]

    style A fill:#1a73e8,color:#fff
    style J fill:#34a853,color:#fff
```

---

## 10. Put-Call Parity, Symmetry, and Static Replication

### 10.1 Put-Call Parity

**Theorem.** For European options on a non-dividend-paying stock:

$$C(S, K, \tau) - P(S, K, \tau) = S - Ke^{-r\tau}$$

**Proof.** Consider two portfolios at time 0:
- Portfolio A: Long one call + invest $Ke^{-r\tau}$ in the bond
- Portfolio B: Long one put + long one share of stock

At maturity $T$:

| | $S_T > K$ | $S_T \le K$ |
|---|---|---|
| Portfolio A | $(S_T - K) + K = S_T$ | $0 + K = K$ |
| Portfolio B | $0 + S_T = S_T$ | $(K - S_T) + S_T = K$ |

Both portfolios have identical terminal values in all states. By no-arbitrage, they must have the same initial value: $C + Ke^{-r\tau} = P + S$. $\blacksquare$

**With continuous dividend yield $\delta$:**

$$C - P = Se^{-\delta\tau} - Ke^{-r\tau}$$

### 10.2 Put-Call Symmetry (Bates, 1997)

Under BS dynamics, a put and call are related by:

$$P(S, K, \sigma, r, \delta) = C(K, S, \sigma, \delta, r) \cdot \frac{K}{S}$$

This means a put on $S$ with strike $K$ is equivalent (up to scaling) to a call on $K$ with strike $S$, with the roles of $r$ and $\delta$ swapped. This symmetry arises from the duality between the physical and numeraire measures.

More precisely, define the log-forward moneyness $m = \ln\frac{F}{K}$ where $F = Se^{(r-\delta)\tau}$:

$$C(m) = P(-m)$$

in terms of normalized (forward) option prices. This is an exact result under log-normal dynamics.

### 10.3 Static Replication

Any European payoff $\Phi(S_T)$ that is twice differentiable can be statically replicated using bonds, forwards, and a continuum of calls and puts (Breeden-Litzenberger, 1978; Carr-Madan, 1998):

$$\Phi(S_T) = \Phi(\kappa) + \Phi'(\kappa)(S_T - \kappa) + \int_0^{\kappa}\Phi''(K)(K - S_T)^+\,dK + \int_{\kappa}^{\infty}\Phi''(K)(S_T - K)^+\,dK$$

for any reference point $\kappa > 0$. The first term is a bond, the second is forwards, and the integrals are puts and calls weighted by the second derivative of the payoff. This decomposition underpins:
- **Variance swap replication:** $\Phi(S_T) = -2\ln(S_T/S_0)$, so $\Phi''(K) = 2/K^2$
- **VIX computation:** Weighted sum of OTM option prices
- **Model-free implied volatility** bounds

---

## 11. Python Implementation

### 11.1 Black-Scholes Pricer with All Greeks

```python
"""
Black-Scholes Option Pricer with Complete Greeks
Module 18 — Quant Nexus Encyclopedia
"""

import numpy as np
from scipy.stats import norm
from typing import NamedTuple

class BSResult(NamedTuple):
    """Complete Black-Scholes result container."""
    price: float
    # First order
    delta: float
    vega: float
    theta: float
    rho: float
    # Second order
    gamma: float
    vanna: float
    volga: float
    charm: float
    veta: float
    # Third order
    speed: float
    color: float
    zomma: float
    ultima: float


def bs_price_greeks(
    S: float, K: float, tau: float, r: float, sigma: float,
    option_type: str = "call"
) -> BSResult:
    """
    Compute BS price and all Greeks for a European option.

    Parameters
    ----------
    S : float — Spot price
    K : float — Strike price
    tau : float — Time to expiry (years)
    r : float — Risk-free rate (continuous)
    sigma : float — Volatility
    option_type : str — "call" or "put"

    Returns
    -------
    BSResult with price and all Greeks through third order.
    """
    assert tau > 0 and sigma > 0 and S > 0 and K > 0

    sqrt_tau = np.sqrt(tau)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * tau) / (sigma * sqrt_tau)
    d2 = d1 - sigma * sqrt_tau

    phi_d1 = norm.pdf(d1)
    Phi_d1 = norm.cdf(d1)
    Phi_d2 = norm.cdf(d2)
    discount = np.exp(-r * tau)

    is_call = option_type.lower() == "call"

    # --- Price ---
    if is_call:
        price = S * Phi_d1 - K * discount * Phi_d2
    else:
        price = K * discount * (1 - Phi_d2) - S * (1 - Phi_d1)

    # --- First Order ---
    delta = Phi_d1 if is_call else Phi_d1 - 1.0
    vega = S * sqrt_tau * phi_d1
    rho_val = (K * tau * discount * Phi_d2) if is_call else (-K * tau * discount * (1 - Phi_d2))
    theta_common = -S * sigma * phi_d1 / (2.0 * sqrt_tau)
    if is_call:
        theta = theta_common - r * K * discount * Phi_d2
    else:
        theta = theta_common + r * K * discount * (1 - Phi_d2)

    # --- Second Order ---
    gamma = phi_d1 / (S * sigma * sqrt_tau)
    vanna = -phi_d1 * d2 / sigma
    volga = vega * d1 * d2 / sigma
    charm_val = -phi_d1 * (2.0 * r * tau - d2 * sigma * sqrt_tau) / (2.0 * tau * sigma * sqrt_tau)
    if not is_call:
        charm_val = charm_val  # Charm for put differs only by the delta drift
    veta_val = vega * (r * d1 / (sigma * sqrt_tau) - (1.0 + d1 * d2) / (2.0 * tau))

    # --- Third Order ---
    speed = -gamma / S * (d1 / (sigma * sqrt_tau) + 1.0)
    color_val = -phi_d1 / (2.0 * S * tau * sigma * sqrt_tau) * (
        1.0 + d1 * (2.0 * r * tau - d2 * sigma * sqrt_tau) / (sigma * sqrt_tau)
    )
    zomma = gamma * (d1 * d2 - 1.0) / sigma
    ultima = -vega / sigma**2 * (
        d1 * d2 * (1.0 - d1 * d2) + d1**2 + d2**2
    )

    return BSResult(
        price=price,
        delta=delta, vega=vega, theta=theta, rho=rho_val,
        gamma=gamma, vanna=vanna, volga=volga, charm=charm_val, veta=veta_val,
        speed=speed, color=color_val, zomma=zomma, ultima=ultima,
    )


# ---------- Quick validation ----------
if __name__ == "__main__":
    res = bs_price_greeks(S=100, K=100, tau=0.25, r=0.05, sigma=0.2, option_type="call")
    print(f"Call Price : {res.price:.6f}")
    print(f"Delta      : {res.delta:.6f}")
    print(f"Gamma      : {res.gamma:.6f}")
    print(f"Vega       : {res.vega:.6f}")
    print(f"Theta      : {res.theta:.6f}")
    print(f"Vanna      : {res.vanna:.6f}")
    print(f"Volga      : {res.volga:.6f}")
    print(f"Speed      : {res.speed:.6f}")
    print(f"Zomma      : {res.zomma:.6f}")
    print(f"Ultima     : {res.ultima:.6f}")
```

### 11.2 Binomial Tree (CRR)

```python
def binomial_tree(
    S: float, K: float, T: float, r: float, sigma: float,
    n: int = 200, option_type: str = "call", american: bool = False
) -> float:
    """
    CRR binomial tree pricer for European and American options.

    Parameters
    ----------
    S, K, T, r, sigma : float — Standard BS parameters
    n : int — Number of time steps
    option_type : str — "call" or "put"
    american : bool — If True, allow early exercise

    Returns
    -------
    float — Option price
    """
    dt = T / n
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    q = (np.exp(r * dt) - d) / (u - d)
    disc = np.exp(-r * dt)

    # Terminal stock prices
    S_T = S * u ** np.arange(n, -1, -1) * d ** np.arange(0, n + 1, 1)

    # Terminal payoffs
    if option_type.lower() == "call":
        payoff = np.maximum(S_T - K, 0.0)
    else:
        payoff = np.maximum(K - S_T, 0.0)

    # Backward induction
    V = payoff.copy()
    for j in range(n - 1, -1, -1):
        V = disc * (q * V[:-1] + (1.0 - q) * V[1:])
        if american:
            S_j = S * u ** np.arange(j, -1, -1) * d ** np.arange(0, j + 1, 1)
            if option_type.lower() == "call":
                intrinsic = np.maximum(S_j - K, 0.0)
            else:
                intrinsic = np.maximum(K - S_j, 0.0)
            V = np.maximum(V, intrinsic)

    return float(V[0])


if __name__ == "__main__":
    # Compare binomial to BS
    bs = bs_price_greeks(100, 100, 0.25, 0.05, 0.2, "call").price
    binom = binomial_tree(100, 100, 0.25, 0.05, 0.2, n=500, option_type="call")
    print(f"\nBS Call     : {bs:.6f}")
    print(f"Binom Call  : {binom:.6f}")
    print(f"Difference  : {abs(bs - binom):.8f}")

    # American put
    am_put = binomial_tree(100, 100, 0.25, 0.05, 0.2, n=500, option_type="put", american=True)
    eu_put = binomial_tree(100, 100, 0.25, 0.05, 0.2, n=500, option_type="put", american=False)
    print(f"\nAmerican Put : {am_put:.6f}")
    print(f"European Put : {eu_put:.6f}")
    print(f"Early Ex Prem: {am_put - eu_put:.6f}")
```

### 11.3 Longstaff-Schwartz LSM for American Options

```python
def lsm_american_put(
    S0: float, K: float, T: float, r: float, sigma: float,
    n_steps: int = 50, n_paths: int = 100_000, n_basis: int = 3,
    seed: int = 42
) -> float:
    """
    Longstaff-Schwartz Least-Squares Monte Carlo for American put.

    Parameters
    ----------
    S0, K, T, r, sigma : float — Standard parameters
    n_steps : int — Number of exercise dates
    n_paths : int — Number of MC paths
    n_basis : int — Number of polynomial basis functions
    seed : int — Random seed

    Returns
    -------
    float — American put price estimate
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    disc = np.exp(-r * dt)

    # Simulate paths using antithetic variates
    half = n_paths // 2
    Z = rng.standard_normal((n_steps, half))
    Z = np.concatenate([Z, -Z], axis=1)  # antithetic

    S = np.zeros((n_steps + 1, n_paths))
    S[0] = S0
    for i in range(n_steps):
        S[i + 1] = S[i] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z[i])

    # Payoff at each step
    payoff = np.maximum(K - S, 0.0)

    # Cash flow matrix: for each path, the (discounted to that time) cash flow
    cashflow = payoff[-1].copy()
    exercise_time = np.full(n_paths, n_steps)

    # Backward induction
    for t in range(n_steps - 1, 0, -1):
        itm = payoff[t] > 0
        if itm.sum() < n_basis + 1:
            cashflow *= disc
            continue

        # Discounted future cash flows for ITM paths
        steps_ahead = exercise_time[itm] - t
        Y = cashflow[itm] * disc ** steps_ahead

        # Regression: use normalized stock price as feature
        X = S[t, itm] / K  # normalize for numerical stability

        # Laguerre-style polynomial basis
        basis = np.column_stack([np.ones_like(X)] +
                                [X**p for p in range(1, n_basis + 1)])

        # Least squares regression
        coeffs, _, _, _ = np.linalg.lstsq(basis, Y, rcond=None)
        continuation = basis @ coeffs

        # Exercise decision
        exercise_value = payoff[t, itm]
        exercise_mask = exercise_value > continuation

        # Update cash flows and exercise times
        itm_indices = np.where(itm)[0]
        exercised = itm_indices[exercise_mask]
        cashflow[exercised] = exercise_value[exercise_mask]
        exercise_time[exercised] = t

    # Discount all cash flows to time 0
    discount_factors = disc ** exercise_time
    price = np.mean(cashflow * discount_factors)

    return price


if __name__ == "__main__":
    am_put_lsm = lsm_american_put(100, 100, 0.25, 0.05, 0.2)
    am_put_tree = binomial_tree(100, 100, 0.25, 0.05, 0.2, n=500,
                                option_type="put", american=True)
    print(f"\nLSM American Put  : {am_put_lsm:.6f}")
    print(f"Tree American Put : {am_put_tree:.6f}")
```

### 11.4 Implied Volatility Solver

```python
def implied_vol(
    price_mkt: float, S: float, K: float, tau: float, r: float,
    option_type: str = "call", tol: float = 1e-12, max_iter: int = 100
) -> float:
    """
    Compute implied volatility via Newton-Raphson with
    Brenner-Subrahmanyam initial guess.

    Parameters
    ----------
    price_mkt : float — Observed market price
    S, K, tau, r : float — Contract and market parameters
    option_type : str — "call" or "put"
    tol : float — Convergence tolerance
    max_iter : int — Maximum iterations

    Returns
    -------
    float — Implied volatility

    Raises
    ------
    ValueError — If no valid implied vol exists or convergence fails
    """
    # Validate bounds
    intrinsic = max(S - K * np.exp(-r * tau), 0.0) if option_type == "call" \
                else max(K * np.exp(-r * tau) - S, 0.0)
    upper_bound = S if option_type == "call" else K * np.exp(-r * tau)

    if price_mkt < intrinsic - 1e-10 or price_mkt > upper_bound + 1e-10:
        raise ValueError(
            f"Market price {price_mkt:.6f} outside no-arbitrage bounds "
            f"[{intrinsic:.6f}, {upper_bound:.6f}]"
        )

    # Brenner-Subrahmanyam initial guess
    sigma = price_mkt / S * np.sqrt(2.0 * np.pi / tau)
    sigma = np.clip(sigma, 0.01, 5.0)

    for i in range(max_iter):
        result = bs_price_greeks(S, K, tau, r, sigma, option_type)
        diff = result.price - price_mkt
        vega = result.vega

        if abs(diff) < tol:
            return sigma

        if abs(vega) < 1e-20:
            # Fallback: bisection step
            if diff > 0:
                sigma *= 0.5
            else:
                sigma *= 2.0
            continue

        # Newton step with damping
        step = diff / vega
        sigma_new = sigma - step
        sigma = np.clip(sigma_new, sigma * 0.1, sigma * 10.0)
        sigma = max(sigma, 1e-6)

    raise ValueError(f"Implied vol did not converge after {max_iter} iterations")


if __name__ == "__main__":
    # Test: compute BS price, then recover implied vol
    true_sigma = 0.25
    test_price = bs_price_greeks(100, 105, 0.5, 0.05, true_sigma, "call").price
    recovered = implied_vol(test_price, 100, 105, 0.5, 0.05, "call")
    print(f"\nTrue sigma     : {true_sigma}")
    print(f"Recovered sigma: {recovered:.12f}")
    print(f"Error          : {abs(true_sigma - recovered):.2e}")
```

---

## 12. C++ Implementation

### 12.1 Optimized Greeks Computation

```cpp
/*
 * Black-Scholes Greeks Engine — Module 18
 * Quant Nexus Encyclopedia
 *
 * Compile: g++ -O3 -march=native -std=c++20 -o bs_greeks bs_greeks.cpp -lm
 * For SIMD: g++ -O3 -mavx2 -mfma -std=c++20 -o bs_greeks bs_greeks.cpp -lm
 */

#include <cmath>
#include <cstdio>
#include <array>
#include <algorithm>
#include <immintrin.h>  // AVX2/FMA intrinsics

// ============================================================
// Fast normal CDF & PDF using rational approximation (Abramowitz & Stegun)
// ============================================================

inline double norm_pdf(double x) {
    constexpr double INV_SQRT_2PI = 0.3989422804014327;
    return INV_SQRT_2PI * std::exp(-0.5 * x * x);
}

inline double norm_cdf(double x) {
    // Hart's approximation — accurate to ~1e-15
    constexpr double a1 =  0.254829592;
    constexpr double a2 = -0.284496736;
    constexpr double a3 =  1.421413741;
    constexpr double a4 = -1.453152027;
    constexpr double a5 =  1.061405429;
    constexpr double p  =  0.3275911;

    int sign = (x >= 0) ? 1 : -1;
    double abs_x = std::fabs(x);
    double t = 1.0 / (1.0 + p * abs_x);
    double poly = t * (a1 + t * (a2 + t * (a3 + t * (a4 + t * a5))));
    double y = 1.0 - poly * std::exp(-0.5 * abs_x * abs_x);

    return 0.5 * (1.0 + sign * y);
}

// ============================================================
// Greeks result structure
// ============================================================

struct BSGreeks {
    double price;
    // First order
    double delta, vega, theta, rho;
    // Second order
    double gamma, vanna, volga, charm, veta;
    // Third order
    double speed, color, zomma, ultima;
};

// ============================================================
// Core computation: all Greeks in a single pass
// ============================================================

BSGreeks bs_greeks(double S, double K, double tau, double r,
                   double sigma, bool is_call = true) {
    BSGreeks g{};

    const double sqrt_tau = std::sqrt(tau);
    const double sig_sqrt = sigma * sqrt_tau;
    const double d1 = (std::log(S / K) + (r + 0.5 * sigma * sigma) * tau) / sig_sqrt;
    const double d2 = d1 - sig_sqrt;
    const double phi_d1 = norm_pdf(d1);
    const double Phi_d1 = norm_cdf(d1);
    const double Phi_d2 = norm_cdf(d2);
    const double disc = std::exp(-r * tau);

    // Price
    if (is_call) {
        g.price = S * Phi_d1 - K * disc * Phi_d2;
    } else {
        g.price = K * disc * (1.0 - Phi_d2) - S * (1.0 - Phi_d1);
    }

    // First order
    g.delta = is_call ? Phi_d1 : (Phi_d1 - 1.0);
    g.vega  = S * sqrt_tau * phi_d1;
    g.rho   = is_call ? (K * tau * disc * Phi_d2)
                       : (-K * tau * disc * (1.0 - Phi_d2));

    double theta_common = -S * sigma * phi_d1 / (2.0 * sqrt_tau);
    g.theta = is_call ? theta_common - r * K * disc * Phi_d2
                      : theta_common + r * K * disc * (1.0 - Phi_d2);

    // Second order
    g.gamma = phi_d1 / (S * sig_sqrt);
    g.vanna = -phi_d1 * d2 / sigma;
    g.volga = g.vega * d1 * d2 / sigma;
    g.charm = -phi_d1 * (2.0 * r * tau - d2 * sig_sqrt)
              / (2.0 * tau * sig_sqrt);
    g.veta  = g.vega * (r * d1 / sig_sqrt - (1.0 + d1 * d2) / (2.0 * tau));

    // Third order
    g.speed = -g.gamma / S * (d1 / sig_sqrt + 1.0);
    g.color = -phi_d1 / (2.0 * S * tau * sig_sqrt)
              * (1.0 + d1 * (2.0 * r * tau - d2 * sig_sqrt) / sig_sqrt);
    g.zomma = g.gamma * (d1 * d2 - 1.0) / sigma;
    g.ultima = -g.vega / (sigma * sigma)
               * (d1 * d2 * (1.0 - d1 * d2) + d1 * d1 + d2 * d2);

    return g;
}

// ============================================================
// SIMD-accelerated batch pricing: 4 options simultaneously (AVX2)
// ============================================================

#ifdef __AVX2__

/*
 * Vectorized norm_pdf for 4 doubles packed in __m256d.
 *
 * SIMD hints for production systems:
 * - Process options in batches of 4 (AVX2) or 8 (AVX-512)
 * - Keep S, K, tau, r, sigma in SoA (struct-of-arrays) layout
 * - Align memory to 32-byte boundaries: alignas(32) double S[N];
 * - Use _mm256_fmadd_pd for fused multiply-add (reduces latency)
 * - Prefetch next batch: _mm_prefetch(&S[i+4], _MM_HINT_T0);
 */

inline __m256d vec_norm_pdf(__m256d x) {
    const __m256d inv_sqrt_2pi = _mm256_set1_pd(0.3989422804014327);
    const __m256d neg_half = _mm256_set1_pd(-0.5);
    __m256d x2 = _mm256_mul_pd(x, x);
    __m256d exponent = _mm256_mul_pd(neg_half, x2);
    // Note: _mm256_exp_pd is not standard; use a polynomial approximation
    // or libmvec for production. Shown conceptually:
    // __m256d result = _mm256_mul_pd(inv_sqrt_2pi, vec_exp(exponent));
    // For this demonstration, we process scalar fallback.
    alignas(32) double vals[4], results[4];
    _mm256_store_pd(vals, x);
    for (int i = 0; i < 4; ++i)
        results[i] = norm_pdf(vals[i]);
    return _mm256_load_pd(results);
}

// Batch compute 4 option prices (same structure, different parameters)
void bs_price_batch(
    const double* __restrict__ S,     // [4]
    const double* __restrict__ K,     // [4]
    const double* __restrict__ tau,   // [4]
    double r, double sigma,
    double* __restrict__ prices       // [4] output
) {
    const __m256d v_r = _mm256_set1_pd(r);
    const __m256d v_sig = _mm256_set1_pd(sigma);
    const __m256d v_half_sig2 = _mm256_set1_pd(0.5 * sigma * sigma);

    __m256d v_S   = _mm256_loadu_pd(S);
    __m256d v_K   = _mm256_loadu_pd(K);
    __m256d v_tau = _mm256_loadu_pd(tau);

    // sqrt(tau)
    __m256d v_sqrt_tau = _mm256_sqrt_pd(v_tau);

    // log(S/K) — scalar fallback for log
    alignas(32) double s_arr[4], k_arr[4], log_sk[4], sqrt_tau_arr[4];
    _mm256_store_pd(s_arr, v_S);
    _mm256_store_pd(k_arr, v_K);
    _mm256_store_pd(sqrt_tau_arr, v_sqrt_tau);

    for (int i = 0; i < 4; ++i)
        log_sk[i] = std::log(s_arr[i] / k_arr[i]);
    __m256d v_log_sk = _mm256_load_pd(log_sk);

    // d1 = (log(S/K) + (r + sig^2/2)*tau) / (sig*sqrt(tau))
    __m256d v_sig_sqrt = _mm256_mul_pd(v_sig, v_sqrt_tau);
    __m256d numerator = _mm256_fmadd_pd(
        _mm256_add_pd(v_r, v_half_sig2), v_tau, v_log_sk);
    __m256d v_d1 = _mm256_div_pd(numerator, v_sig_sqrt);
    __m256d v_d2 = _mm256_sub_pd(v_d1, v_sig_sqrt);

    // CDF and exp — scalar fallback
    alignas(32) double d1_arr[4], d2_arr[4], tau_arr[4];
    _mm256_store_pd(d1_arr, v_d1);
    _mm256_store_pd(d2_arr, v_d2);
    _mm256_store_pd(tau_arr, v_tau);

    for (int i = 0; i < 4; ++i) {
        double Phi_d1 = norm_cdf(d1_arr[i]);
        double Phi_d2 = norm_cdf(d2_arr[i]);
        double disc = std::exp(-r * tau_arr[i]);
        prices[i] = s_arr[i] * Phi_d1 - k_arr[i] * disc * Phi_d2;
    }
}

#endif  // __AVX2__

// ============================================================
// CRR Binomial Tree
// ============================================================

double binomial_tree(double S, double K, double T, double r,
                     double sigma, int n, bool is_call, bool american) {
    const double dt = T / n;
    const double u  = std::exp(sigma * std::sqrt(dt));
    const double d  = 1.0 / u;
    const double q  = (std::exp(r * dt) - d) / (u - d);
    const double disc = std::exp(-r * dt);

    // Use a single vector for the price lattice (memory efficient)
    std::vector<double> V(n + 1);

    // Terminal payoffs
    for (int i = 0; i <= n; ++i) {
        double S_T = S * std::pow(u, n - i) * std::pow(d, i);
        V[i] = is_call ? std::max(S_T - K, 0.0) : std::max(K - S_T, 0.0);
    }

    // Backward induction
    for (int j = n - 1; j >= 0; --j) {
        for (int i = 0; i <= j; ++i) {
            V[i] = disc * (q * V[i] + (1.0 - q) * V[i + 1]);
            if (american) {
                double S_node = S * std::pow(u, j - i) * std::pow(d, i);
                double intrinsic = is_call ? std::max(S_node - K, 0.0)
                                           : std::max(K - S_node, 0.0);
                V[i] = std::max(V[i], intrinsic);
            }
        }
    }
    return V[0];
}

// ============================================================
// Main: demonstration and validation
// ============================================================

int main() {
    // Test parameters
    double S = 100.0, K = 100.0, tau = 0.25, r = 0.05, sigma = 0.20;

    auto g = bs_greeks(S, K, tau, r, sigma, true);

    std::printf("=== Black-Scholes Greeks (ATM Call) ===\n");
    std::printf("Price  : %12.6f\n", g.price);
    std::printf("Delta  : %12.6f\n", g.delta);
    std::printf("Gamma  : %12.6f\n", g.gamma);
    std::printf("Vega   : %12.6f\n", g.vega);
    std::printf("Theta  : %12.6f\n", g.theta);
    std::printf("Rho    : %12.6f\n", g.rho);
    std::printf("Vanna  : %12.6f\n", g.vanna);
    std::printf("Volga  : %12.6f\n", g.volga);
    std::printf("Charm  : %12.6f\n", g.charm);
    std::printf("Speed  : %12.6f\n", g.speed);
    std::printf("Color  : %12.6f\n", g.color);
    std::printf("Zomma  : %12.6f\n", g.zomma);
    std::printf("Ultima : %12.6f\n", g.ultima);

    // Binomial tree comparison
    double binom_eu = binomial_tree(S, K, tau, r, sigma, 500, true, false);
    double binom_am_put = binomial_tree(S, K, tau, r, sigma, 500, false, true);
    double binom_eu_put = binomial_tree(S, K, tau, r, sigma, 500, false, false);

    std::printf("\n=== Binomial Tree (n=500) ===\n");
    std::printf("EU Call     : %12.6f (BS: %12.6f)\n", binom_eu, g.price);
    std::printf("AM Put      : %12.6f\n", binom_am_put);
    std::printf("EU Put      : %12.6f\n", binom_eu_put);
    std::printf("Early Ex    : %12.6f\n", binom_am_put - binom_eu_put);

    return 0;
}
```

---

## 13. Exercises

### Problem 1: Risk-Neutral Probability
A stock is at $S_0 = 50$. In one period, it can go to $60$ or $42$. The risk-free rate is 5% per period. (a) Compute the risk-neutral probability $q$. (b) Price a European call with strike $K = 48$. (c) Verify that $q$ makes the discounted stock a martingale.

### Problem 2: CRR Convergence
Implement a two-period CRR tree for $S_0 = 100$, $K = 100$, $T = 1$, $r = 0.06$, $\sigma = 0.30$. (a) Compute the European call price by backward induction. (b) Compute the price using the direct binomial summation formula. (c) Compare to the Black-Scholes price. (d) Plot the convergence of the binomial price to BS as $n \to \infty$.

### Problem 3: Deriving the PDE
Starting from the dynamics $dS = \mu S\,dt + \sigma S\,dW$, derive the Black-Scholes PDE for a derivative $V(S, t)$. Explicitly show: (a) the application of Ito's lemma, (b) the construction of the delta-hedged portfolio, (c) the elimination of randomness, and (d) the application of the no-arbitrage condition.

### Problem 4: Heat Equation
Verify that the transformation $V(S,t) = Ke^{\beta x + \gamma \tau}u(x,\tau)$ with the substitutions given in Section 2.4 reduces the BS PDE to the heat equation. Find the explicit values of $\beta$ and $\gamma$ in terms of $r$ and $\sigma$.

### Problem 5: Greek Verification
For $S = 100$, $K = 95$, $\tau = 0.5$, $r = 0.03$, $\sigma = 0.25$: (a) Compute all first- and second-order Greeks analytically. (b) Verify delta, gamma, and vega using finite differences with step sizes $h = 1, 0.1, 0.01, 0.001$. (c) Demonstrate that the BS PDE relation $\Theta + \frac{1}{2}\sigma^2 S^2\Gamma + rS\Delta - rC = 0$ holds numerically.

### Problem 6: Implied Volatility Surface
Using the Python code from Section 11.4, construct an implied volatility surface: (a) Generate BS call prices for $K \in \{80, 85, \ldots, 120\}$ and $\tau \in \{0.1, 0.25, 0.5, 1.0\}$ using $\sigma = 0.20$. (b) Add noise: replace each price with $C + \epsilon$ where $\epsilon \sim \mathcal{N}(0, 0.05)$. (c) Recover implied vols and plot the surface. (d) Explain why the recovered surface is no longer flat.

### Problem 7: Barrier Option
For an up-and-out call with $S = 100$, $K = 90$, $H = 120$, $r = 0.05$, $\sigma = 0.30$, $T = 1$: (a) Price analytically using the reflection formula. (b) Price via Monte Carlo with 500,000 paths. (c) Explain why the MC price may differ from the analytic price (hint: barrier monitoring frequency).

### Problem 8: American Put Pricing
For $S = 36$, $K = 40$, $T = 1$, $r = 0.06$, $\sigma = 0.20$: (a) Price the American put using a 1000-step binomial tree. (b) Price using LSM with 200,000 paths and 50 exercise dates. (c) Estimate the early exercise boundary $S^*(t)$ for $t \in \{0, 0.1, 0.2, \ldots, 1.0\}$ from the binomial tree. (d) Plot the early exercise boundary and interpret its shape.

### Problem 9: Put-Call Parity Arbitrage
A European call on a non-dividend-paying stock is quoted at $C = 12.50$ with $S = 100$, $K = 92$, $T = 0.5$, $r = 0.04$. The corresponding put is quoted at $P = 3.20$. (a) Check whether put-call parity holds. (b) If violated, construct an arbitrage portfolio and compute the risk-free profit. (c) Discuss why such violations may persist momentarily in practice (transaction costs, bid-ask spreads).

### Problem 10: Exotic Portfolio Greeks
Consider a portfolio consisting of: 100 vanilla calls ($K = 100$), short 50 up-and-out calls ($K = 100$, $H = 130$), and 30 Asian calls (arithmetic average, $K = 100$). All on the same underlying with $S = 100$, $\sigma = 0.25$, $r = 0.05$, $T = 0.5$. (a) Estimate the portfolio price by Monte Carlo. (b) Compute the portfolio delta and gamma by bumping $S$ by $\pm 1$. (c) Compute vega by bumping $\sigma$ by $\pm 0.01$. (d) Discuss the challenges of hedging this portfolio compared to a plain vanilla book.

---

## 14. Summary and Bridge to Module 19

### Key Results

This module established the complete mathematical framework for derivatives pricing in the Black-Scholes world:

- **The binomial model** provides a discrete, intuitive foundation. Risk-neutral pricing emerges from no-arbitrage, not from assumptions about investor preferences. The CRR parameterization converges to Black-Scholes as the number of periods grows.

- **The Black-Scholes PDE** follows from the delta-hedging argument and Ito's lemma. The drift $\mu$ of the underlying disappears from the equation --- a profound result that underlies risk-neutral pricing.

- **The risk-neutral (martingale) approach** provides an alternative derivation: under the equivalent martingale measure $\mathbb{Q}$, discounted prices are martingales and option prices are discounted expected payoffs.

- **The Greeks** form a complete sensitivity hierarchy. First-order Greeks (delta, vega, theta, rho) drive daily risk management. Second-order Greeks (gamma, vanna, volga) are critical for understanding P&L variance and volatility risk. Third-order Greeks (speed, color, zomma, ultima) matter for large portfolios and exotic books.

- **Implied volatility** inverts the BS formula and reveals market-implied distributional assumptions. The smile/skew exposes the failure of constant-volatility log-normal assumptions.

- **American options** require solving optimal stopping problems. The Longstaff-Schwartz algorithm provides a practical MC-based solution via regression on continuation values.

- **Put-call parity** is a model-free no-arbitrage relation; static replication decomposes arbitrary payoffs into vanilla options.

### What Black-Scholes Cannot Do

The constant-volatility assumption is the model's central limitation. In reality:
- Volatility is stochastic and mean-reverting
- Volatility is negatively correlated with returns (leverage effect)
- Returns exhibit jumps
- The risk-neutral density has fatter tails than log-normal

### Bridge to Module 19

**Module 19 (Local & Stochastic Volatility Models)** addresses these limitations directly:

- **Local volatility** (Dupire, 1994): $\sigma(S, t)$ is a deterministic function calibrated to match the entire implied volatility surface. We will derive Dupire's formula $\sigma_{\text{loc}}^2(K,T) = \frac{\partial C/\partial T + rK\,\partial C/\partial K}{\frac{1}{2}K^2\,\partial^2 C/\partial K^2}$.

- **Stochastic volatility** (Heston, 1993; SABR, 2002): Volatility follows its own SDE. This generates smile dynamics that local vol cannot.

- **Jump-diffusion** (Merton, 1976): Adds Poisson jumps to the diffusion, producing short-dated smile effects.

The Greeks framework developed here extends to all these models --- though closed-form expressions give way to numerical computation (finite differences, AAD; see Module 30).

---

*Next: [Module 19 — Stochastic Volatility Models](../Asset_Pricing/19_stochastic_volatility.md)*

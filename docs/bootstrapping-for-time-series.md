# Bootstrapping for Time Series Models

This document explains bootstrapping techniques, with a focus on their application to time series data and models.

## Table of Contents

1. [Introduction to Bootstrapping](#introduction-to-bootstrapping)
2. [Classical Bootstrap](#classical-bootstrap)
3. [Why Bootstrap Fails for Time Series](#why-bootstrap-fails-for-time-series)
4. [Block Bootstrap Methods](#block-bootstrap-methods)
5. [Choosing Block Length](#choosing-block-length)
6. [Practical Considerations](#practical-considerations)

## Introduction to Bootstrapping

Bootstrapping is a resampling method used to estimate the sampling distribution of a statistic by repeatedly sampling
from the observed data. It allows us to quantify uncertainty without making strong parametric assumptions.

### Key Concepts

- **Purpose**: Estimate confidence intervals, standard errors, and assess model stability
- **Principle**: The empirical distribution approximates the true population distribution
- **Method**: Generate many pseudo-datasets by resampling from the original data

## Classical Bootstrap

The classical bootstrap, introduced by Efron (1979), works well for independent and identically distributed (i.i.d.)
data.

### Algorithm

1. Start with original dataset of size n: [x₁, x₂, ..., xₙ]
2. Draw n samples with replacement to create a bootstrap sample
3. Calculate the statistic of interest on the bootstrap sample
4. Repeat steps 2-3 many times (typically B = 1000-10000)
5. Use the distribution of bootstrap statistics to estimate uncertainty

### Illustration: Classical Bootstrap

```
Original Data (n=8):
[A, B, C, D, E, F, G, H]

Bootstrap Sample 1 (random draw with replacement):
[A, C, C, B, H, A, E, G]
         ^     ^
      repeated elements

Bootstrap Sample 2:
[D, D, F, A, B, C, E, H]
   ^
repeated

Bootstrap Sample 3:
[B, A, E, E, C, F, F, A]
         ^     ^     ^
      repeated elements

... (repeat B times)
```

### When Classical Bootstrap Works

```
Independent Data Points:
○   ○   ○   ○   ○   ○   ○   ○
|   |   |   |   |   |   |   |
No temporal dependence - each observation is independent

Resampling doesn't break any structure because there is none to preserve
```

## Why Bootstrap Fails for Time Series

Time series data has **temporal dependence** - observations close in time are correlated. Classical bootstrap destroys
this structure.

### The Problem

```
Original Time Series with Autocorrelation:
Time:  t₁   t₂   t₃   t₄   t₅   t₆   t₇   t₈
Data:  [5] -[6] -[7] -[4] -[5] -[8] -[9] -[8]
        └───┘   └───┘   └───┘   └───┘
      correlated  pairs  show  trend

Classical Bootstrap Sample (WRONG):
[t₇, t₂, t₂, t₅, t₈, t₁, t₃, t₆]
 [9,  6,  6,  5,  8,  5,  7,  8]
      ^           ^
  time order destroyed - temporal structure lost!

Result: Underestimates true variability and breaks dependencies
```

### Consequences

1. **Underestimated variance**: Breaks correlation structure that contributes to uncertainty
2. **Invalid inference**: Confidence intervals too narrow
3. **Loss of dynamics**: Autocorrelation, trends, and seasonality disappear

## Block Bootstrap Methods

Block bootstrap methods preserve temporal dependence by resampling contiguous blocks rather than individual
observations.

### Moving Block Bootstrap (MBB)

**Concept**: Create overlapping blocks of length ℓ and resample these blocks.

```
Original Time Series (n=12):
Time:  1   2   3   4   5   6   7   8   9   10  11  12
Data: [A] -[B] -[C] -[D] -[E] -[F] -[G] -[H] -[I] -[J] -[K] -[L]

Block Length ℓ=4, create overlapping blocks:
Block 1:  [A, B, C, D]
Block 2:     [B, C, D, E]
Block 3:        [C, D, E, F]
Block 4:           [D, E, F, G]
Block 5:              [E, F, G, H]
Block 6:                 [F, G, H, I]
Block 7:                    [G, H, I, J]
Block 8:                       [H, I, J, K]
Block 9:                          [I, J, K, L]

Bootstrap Sample (randomly select blocks):
Block 3 + Block 7 + Block 2 =
[C, D, E, F] + [G, H, I, J] + [B, C, D, E]

Result: [C, D, E, F, G, H, I, J, B, C, D, E]
         └─────┘ └─────┘ └─────┘
      temporal structure preserved within blocks
```

### Circular Block Bootstrap (CBB)

**Concept**: Treat the time series as circular to create exactly n/ℓ non-overlapping blocks without edge effects.

```
Original Time Series (n=12, ℓ=4):
[A] -[B] -[C] -[D] -[E] -[F] -[G] -[H] -[I] -[J] -[K] -[L]
                                                           └──wraps──┐
                                                                     ↓
Circular blocks:                                                    [A]
Block 1:  [A, B, C, D]
Block 2:  [E, F, G, H]
Block 3:  [I, J, K, L]
Block 4:  [K, L, A, B]  (wraps around)
Block 5:  [G, H, I, J]
...

Bootstrap Sample: randomly select and concatenate blocks
```

### Stationary Bootstrap (SB)

**Concept**: Use random block lengths following a geometric distribution to better preserve stationarity.

```
Original Time Series:
[A] -[B] -[C] -[D] -[E] -[F] -[G] -[H] -[I] -[J]

Block lengths drawn from Geometric(p):
Expected length = 1/p

Bootstrap Sample with variable blocks:
[C, D] + [F, G, H, I, J] + [A] + [D, E, F, G]
  ↑          ↑              ↑         ↑
 ℓ=2        ℓ=5            ℓ=1       ℓ=4

Advantage: Reduces bias from fixed block length
```

### Seasonal Block Bootstrap

**Concept**: For seasonal data, use blocks of length equal to or multiple of the seasonal period.

```
Monthly Data with Annual Seasonality (period=12):

Original (3 years = 36 months):
Year 1: [J, F, M, A, M, J, J, A, S, O, N, D]
Year 2: [J, F, M, A, M, J, J, A, S, O, N, D]
Year 3: [J, F, M, A, M, J, J, A, S, O, N, D]

Block Length ℓ=12 (one season):
Block 1: [Year 1]
Block 2: [Year 2]
Block 3: [Year 3]

Bootstrap Sample: [Year 2] + [Year 1] + [Year 2]
Preserves seasonal patterns within each year
```

## Choosing Block Length

The block length ℓ is critical: too small destroys dependence, too large reduces diversity.

### Trade-off Visualization

```
Block Length Spectrum:

ℓ=1 (Classical Bootstrap)
├─┼─┼─┼─┼─┼─┼─┼─┤
Pro: Maximum diversity
Con: Destroys all temporal structure
Result: Bias HIGH, Variance LOW

ℓ=moderate (Optimal)
├────┼────┼────┼────┤
Pro: Preserves short-term dependence
Pro: Sufficient diversity
Result: Bias MEDIUM, Variance MEDIUM

ℓ=n (No Bootstrap)
├─────────────────┤
Pro: Perfect temporal structure
Con: No resampling diversity
Result: Bias LOW, Variance HIGH
```

### Selection Guidelines

**Rule of Thumb**: For autocorrelation dying out at lag k, use ℓ ≈ k^(1/3) × n^(1/3)

**Practical Approach**:
1. Plot autocorrelation function (ACF)
2. Find lag where ACF drops below significance threshold
3. Set block length to 1.5-2 times this lag

```
ACF Plot Example:

 ACF
 1.0 |█
 0.8 |███
 0.6 |████
 0.4 |█████ ──── significance threshold (e.g., 0.4)
 0.2 |██████
 0.0 |███████
-0.2 |────────
     └────────────────────
      1  2  3  4  5  6  7  Lag

Significant autocorrelation up to lag 5
→ Choose block length ℓ = 8-10
```

## Practical Considerations

### Number of Bootstrap Samples

- **B = 1000**: Sufficient for standard error estimation
- **B = 2000-5000**: Better for confidence intervals
- **B = 10000+**: Required for extreme quantiles or small p-values

### Overlapping vs Non-overlapping Blocks

```
Non-overlapping Blocks (ℓ=4, n=12):
[A B C D] [E F G H] [I J K L]
   └─3 blocks only─┘

Overlapping Blocks (ℓ=4, n=12):
[A B C D]
  [B C D E]
    [C D E F]
      [D E F G]
        [E F G H]
          [F G H I]
            [G H I J]
              [H I J K]
                [I J K L]
   └─9 blocks available─┘

Overlapping provides more bootstrap samples but introduces dependence between blocks
```

### Model-Based vs Residual Bootstrap

**Residual Bootstrap** (for fitted models):

```
1. Fit model to original data:
   Observed: Y(t) = f(X(t)) + ε(t)
   Fitted:   Ŷ(t) = f(X(t))
   Residual: ε̂(t) = Y(t) - Ŷ(t)

2. Bootstrap residuals in blocks:
   [ε̂₃, ε̂₄, ε̂₅] + [ε̂₁, ε̂₂, ε̂₃] + ...

3. Create bootstrap response:
   Y*(t) = Ŷ(t) + ε̂*(t)

4. Refit model to Y*(t)

Advantage: Preserves structure in predictors while allowing residual dependence
```

### Implementation Example Workflow

```
Time Series Validation Pipeline:

Original Data
     ↓
[1] Fit Model
     ↓
[2] Extract Residuals
     ↓
[3] Determine Block Length
     ↓
[4] Generate B Bootstrap Samples
     ↓
     ├─→ Sample 1 → Fit Model → Prediction₁
     ├─→ Sample 2 → Fit Model → Prediction₂
     ├─→ Sample 3 → Fit Model → Prediction₃
     └─→ ...
     ↓
[5] Aggregate Predictions
     ↓
[6] Calculate Confidence Intervals
     ↓
Results: Mean ± CI
```

## References

- Efron, B. (1979). Bootstrap methods: Another look at the jackknife.
- Künsch, H.R. (1989). The jackknife and the bootstrap for general stationary observations.
- Politis, D.N. & Romano, J.P. (1994). The stationary bootstrap.
- Lahiri, S.N. (2003). Resampling Methods for Dependent Data.

## Application to Climate Models

For climate forecasting models like those in chapkit:

1. **Use seasonal block bootstrap** for monthly/seasonal climate data
2. **Set block length** to capture ENSO cycles (12-18 months)
3. **Apply residual bootstrap** when validating fitted prediction models
4. **Generate prediction intervals** using bootstrap distribution of forecasts
5. **Assess model stability** by examining variability of parameters across bootstrap samples

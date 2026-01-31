# What Has a Foundation Model Found? Using Inductive Bias to Probe for World Models

## Executive Summary

This paper introduces the **inductive bias probe**, a framework for evaluating whether foundation models have learned postulated world models or merely task-specific heuristics. The probe measures a model's extrapolative behavior when adapted to small synthetic datasets consistent with a hypothesized world model. Across three domains—orbital mechanics, lattice navigation, and the game of Othello—the authors demonstrate that foundation models achieve high next-token prediction accuracy (R² > 0.9999 for orbital trajectories, >99% legal move prediction for Othello) while exhibiting weak inductive bias toward the underlying world models. Specifically, transformer-based models trained on orbital trajectories fail to recover Newtonian mechanics when fine-tuned to predict force vectors; symbolic regression recovers nonsensical laws such as 
```
F ∝ (sin(1/sin(r-0.24)) + 1.45) * 1/(1/r + m2)
```
rather than 
```
F ∝ m1m2/r^2
```
The analysis reveals that models develop inductive biases toward **legal next-token partitions** (coarsened state representations that preserve valid actions) rather than true state structures. State-space models (Mamba, Mamba-2) and LSTMs consistently outperform transformers in inductive bias metrics for lattice problems, though all architectures show degradation as state space complexity increases. These findings challenge the assumption that sequence prediction competency implies latent world model acquisition.

---

## 1. Technical Architecture

### 1.1 Motivation and Problem Statement

Foundation models are presumed to uncover deeper domain understanding through sequence prediction. However, standard evaluation conflates predictive accuracy with world model learning. The core problem is distinguishing between:
- **World model learning**: The model extrapolates based on underlying state structure (e.g., Newtonian state vectors for orbital mechanics)
- **Task-specific heuristics**: The model learns input-output mappings that generalize poorly to novel functions of state

Given a world model defined by state space Φ and mapping φ: X → Φ, the challenge is determining whether a foundation model's inductive bias aligns with functions consistent with Φ or relies on spurious correlations.

### 1.2 Architecture Components

**Inductive Bias Probe**: A procedure that repeatedly fits a foundation model to synthetic datasets D ∼ P_D consistent with a postulated world model and compares the learned functions to those allowed by the world model.

**Core Metrics**:

**R-IB (Respecting State)**:
```
R-IB = E[1(𝑚̂_D(X_i), 𝑚̂_D(X_j)) | φ(X_i) = φ(X_j)]
```
Measures similarity of predictions for inputs mapping to identical states. Range [0,1], where 1 indicates perfect state respect.

**D-IB (Distinguishing State)**:
```
D-IB = 1 - E[1(𝑚̂_D(X_i), 𝑚̂_D(X_j)) | φ(X_i) ≠ φ(X_j)]
```
Measures dissimilarity of predictions for inputs mapping to different states. Range [0,1], where 1 indicates perfect state distinction.

**Extrapolative Predictability** (continuous extension):
```
Î(x_i, x_j) = -min_{h∈H} E_D[ℓ(h(𝑚̂_D(x_i)), 𝑚̂_D(x_j))]
```
Compared against oracle extrapolative predictability I* computed from true state-based extrapolation.

**Next-Token Coarsening Metric**:
Decomposes D-IB into:
- D-IB_q= : Predictability for distinct states with identical legal next tokens
- D-IB_q≠ : Predictability for distinct states with different legal next tokens

If D-IB_q= < D-IB_q≠, the model biases toward next-token partitions rather than true state.

### 1.3 Training or Pre-Training Protocol

**Models Evaluated**:
- **Transformer**: 109M parameters, 12 layers, 12 attention heads, 768 embedding dimensions
- **RNN**: 6 unidirectional layers, 768 embedding dimensions (2 layers for lattice)
- **LSTM**: Same architecture as RNN with LSTM layers
- **Mamba**: 24 layers (analogous to 12 transformer layers), 768 embedding dimensions, SSM state expansion factor 16
- **Mamba-2**: Same architecture as Mamba with Mamba-2 mixer modules

**Optimization**:
- Optimizer: Adam (learning rate 6e-4, 2000 warmup iterations)
- Weight decay: 0.1
- Gradient clipping: 1.0
- Loss: Cross-entropy for discrete coordinates, MSE for continuous (physics force prediction)

**Pre-training Datasets**:
1. **Orbital Mechanics**: 10M sequences (20B tokens) of planetary trajectories simulated via Newtonian mechanics. Sequences encode (x,y) coordinates across time intervals (6-month or 1-week), discretized into 7K bins per coordinate.
2. **Lattice**: 10M tokens from random walks on line segments with S ∈ {2,3,4,5} states. Vocabulary: {L, ⊥, R} (left, stay, right).
3. **Othello**: 20M games (3.8M hold-out), tokenized as sequences of board positions (64 possible squares).

### 1.4 Performance Impact

**Next-Token Prediction Performance**:
- Orbital mechanics: R² > 0.9999, MSE at 1-step prediction: (1.90±0.45)·10^-8 vs. baseline (1.16±0.21)·10^-4
- Othello: >99% of top predictions are legal moves across all architectures
- Lattice: 100% legal move prediction for 5-state lattices

**Computational Efficiency**:
Symbolic regression (PySR) for law discovery constrained to:
- Max expression size: 20
- Binary operators: {+, ×}
- Unary operators: {sin, cos, exp, inverse}
- Iterations: 100 per restart, 3 random restarts

---

## 2. Post-Training or Optimization Methods

**Fine-tuning for Probe Evaluation**:
Models are fine-tuned on 100 synthetic datasets, each containing 100 examples. For each dataset:
1. Sample sequences uniformly at random
2. Assign outputs (binary for lattice/Othello, continuous for physics) consistent with world model state
3. Fine-tune for state prediction (not next-token)
4. Extract prediction functions across hold-out inputs

**Force Vector Prediction** (Physics):
- Fine-tune on 1% of force vector observations from solar system trajectories
- Normalize force vectors per sequence to unit maximum length
- Learning rate grid search: [1e-6, 5e-4], selected 2e-4
- Training steps: 10,000
- Evaluation: MSE on held-out trajectory points

**Symbolic Regression**:
Applied to force magnitude predictions to recover implied physical laws:
- Training: 9K two-body problems
- Test: 1K sequences, selecting 5,000 timesteps with states nearest to training distribution
- Distance metric: Euclidean distance in true 6D state space (masses, relative positions, relative velocities)

---

## 3. Agentic or System-Level Design

Not applicable per source document.

---

## 4. Benchmark Performance and Ablations

**Table 1: Inductive Bias Metrics (Lattice 5-State and Othello)**

| Architecture | Lattice R-IB (↑) | Lattice D-IB (↑) | Othello R-IB (↑) | Othello D-IB (↑) |
|-------------|-----------------|-----------------|-----------------|-----------------|
| RNN | 0.574 (0.026) | 0.803 (0.032) | 0.632 (0.023) | 0.797 (0.023) |
| LSTM | 0.782 (0.021) | 0.921 (0.030) | 0.563 (0.030) | 0.610 (0.034) |
| Transformer | 0.483 (0.031) | 0.677 (0.034) | 0.703 (0.025) | 0.624 (0.033) |
| Mamba | 0.571 (0.023) | 0.866 (0.029) | 0.682 (0.021) | 0.728 (0.027) |
| Mamba-2 | 0.617 (0.021) | 0.864 (0.029) | 0.653 (0.022) | 0.694 (0.029) |

*Values shown are means (standard errors). ↑ indicates higher is better.*

**Table 2: Recovered Force Laws (Physics)**

| Source | Recovered Law |
|--------|--------------|
| True (Newton) | `F ∝ m1*m2/r^2` |
| Transformer (Galaxy 1) | `F ∝ (sin(1/sin(r-0.24)) + 1.45) * 1/(1/r + m2)` |
| Transformer (Galaxy 3) | `F ∝ cos(sin(0.48)) * m1` |
| Transformer (Galaxy 4) | `F ∝ sin(r + 8569.2) + 1/m1` |
| Transformer (Galaxy 5) | `F ∝ cos(cos(e^m2))` |
| Oracle (k-NN on true state) | `F ∝ m1*m2/r^2` |

*Symbolic regression applied to force magnitude predictions. Transformer recovers different nonsensical laws for different data samples; oracle consistently recovers ground truth.*

**Table 3: Orbital Trajectory Prediction Accuracy**

| Model | 1-step MSE | 5-step MSE | 100-step MSE |
|-------|-----------|-----------|-------------|
| Per-orbit mean | (7.53±0.59)·10^-2 | (5.53±0.58)·10^-2 | (1.39±0.08)·10^-1 |
| Previous position | (1.16±0.21)·10^-4 | (1.37±0.38)·10^-4 | (4.04±0.47)·10^-2 |
| Transformer | (1.90±0.45)·10^-8 | (1.56±0.45)·10^-8 | (3.74±3.37)·10^-5 |

*MSE computed on held-out trajectories. Transformer achieves near-perfect next-step prediction despite lacking Newtonian inductive bias.*

**Table 4: Next-Token Legality Test**

| Architecture | Lattice (5 states) | Othello |
|-------------|-------------------|---------|
| RNN | 1.00 | 0.992 |
| LSTM | 1.00 | 0.996 |
| Transformer | 1.00 | 0.999 |
| Mamba | 1.00 | 0.999 |
| Mamba-2 | 1.00 | 0.999 |

*Fraction of top predictions that are legal moves in the true underlying state.*

**Table 5: Transfer Learning Performance (Othello)**

| Architecture | Pre-training | Majority Tiles ACC (↑) | Board Balance ACC (↑) | Edge Balance ACC (↑) |
|-------------|-------------|----------------------|---------------------|-------------------|
| RNN | Untrained | 0.755 (0.003) | 0.806 (0.003) | 0.816 (0.002) |
| RNN | NTP trained | 0.792 (0.002) | 0.856 (0.002) | 0.964 (0.001) |
| LSTM | Untrained | 0.786 (0.003) | 0.864 (0.002) | 0.953 (0.001) |
| LSTM | NTP trained | 0.901 (0.002) | 0.927 (0.001) | 0.982 (0.001) |
| Transformer | Untrained | 0.754 (0.003) | 0.855 (0.002) | 0.967 (0.001) |
| Transformer | NTP trained | 0.956 (0.001) | 0.965 (0.001) | 0.996 (0.000) |
| Mamba | Untrained | 0.816 (0.002) | 0.888 (0.002) | 0.952 (0.001) |
| Mamba | NTP trained | 0.937 (0.002) | 0.931 (0.002) | 0.989 (0.001) |
| Mamba-2 | Untrained | 0.821 (0.002) | 0.891 (0.002) | 0.969 (0.001) |
| Mamba-2 | NTP trained | 0.970 (0.001) | 0.976 (0.001) | 0.995 (0.001) |

*Transfer tasks: Majority Tiles (which color dominates), Board Balance (top vs. bottom half), Edge Balance (edge vs. center). IB Correlation with R-IB/(1-D-IB): 0.462-0.970 across tasks.*

**Table 6: Decomposition of D-IB by Next-Token Equivalence**

| Architecture | Lattice D-IB_q= | Lattice D-IB_q≠ | Othello D-IB_q= | Othello D-IB_q≠ |
|-------------|----------------|----------------|----------------|----------------|
| RNN | 0.740 (0.042) | 0.844 (0.034) | 0.521 (0.031) | 0.798 (0.023) |
| LSTM | 0.873 (0.051) | 0.952 (0.034) | 0.519 (0.035) | 0.610 (0.034) |
| Transformer | 0.626 (0.037) | 0.710 (0.037) | 0.458 (0.033) | 0.625 (0.033) |
| Mamba | 0.764 (0.040) | 0.933 (0.035) | 0.485 (0.030) | 0.729 (0.027) |
| Mamba-2 | 0.778 (0.042) | 0.920 (0.033) | 0.553 (0.032) | 0.694 (0.029) |

*D-IB_q= measures predictability for distinct states with identical legal next tokens; D-IB_q≠ for distinct states with different legal next tokens. Lower D-IB_q= indicates bias toward next-token partitions.*

---

## 5. Key Technical Takeaways

- High next-token prediction accuracy (R² > 0.9999 for orbital mechanics, 99.9% legality for Othello) does not imply learning of underlying world models
- Transformer architectures exhibit weaker inductive bias toward state structure compared to recurrent (LSTM) and state-space (Mamba) architectures in lattice navigation tasks
- Foundation models trained on orbital trajectories recover inconsistent, non-physical force laws when fine-tuned on force prediction; symbolic regression yields equations such as `F ∝ sin(1/sin(r-0.24))` rather than Newton's law
- Models demonstrate inductive bias toward **legal next-token partitions** (coarsened state representations) rather than true state structure, as evidenced by D-IB_q= < D-IB_q≠ across all architectures
- Inductive bias metrics (R-IB, D-IB) correlate strongly with transfer learning performance (correlation 0.462-0.970) on downstream tasks requiring true state understanding
- Performance on inductive bias probes degrades as state space complexity increases (tested up to 5 states for lattice problems)

---

## 6. Conclusion

The inductive bias probe framework demonstrates that foundation models can achieve expert-level sequence prediction while failing to acquire the underlying world models governing the data generating process. Rather than learning compact, generalizable representations (e.g., Newtonian state vectors), models develop task-specific heuristics and coarsened state representations that preserve legal action sets but discard information necessary for novel state-dependent tasks. This discrepancy is particularly pronounced in transformer architectures compared to state-space models. The findings indicate that next-token prediction optimization does not automatically induce world model learning, suggesting that explicit inductive biases or architectural constraints may be necessary for foundation models to achieve the "Newtonian leap" from pattern recognition to mechanistic understanding.

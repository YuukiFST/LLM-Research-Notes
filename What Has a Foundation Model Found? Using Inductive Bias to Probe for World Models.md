# What Has a Foundation Model Found? Using Inductive Bias to Probe for World Models

## Executive Summary

This paper introduces the inductive bias probe, a framework for evaluating whether foundation models trained on sequence prediction develop coherent world models or merely task-specific heuristics. The methodology repeatedly applies foundation models to synthetic datasets consistent with a postulated world model and measures whether extrapolations align with that model's state structure. Empirical results across three domains (orbital mechanics, lattice problems, Othello) reveal that models achieving near-perfect next-token prediction accuracy (>99%) nonetheless fail to develop inductive biases toward underlying world models. In orbital mechanics, transformers trained on 10M trajectories predict future positions with R² > 0.9999 but recover nonsensical force laws (e.g., F ∝ sin(sin(1/sin(r-0.24))+1.45)×(1/(1/r)+m₂)) when fine-tuned on force vectors, differing fundamentally from Newton's inverse-square law. Analysis reveals models develop coarsened representations based on legal next-token partitions rather than true state spaces, with D-IB_q= consistently lower than D-IB_q≠ across architectures.

---

## 1. Technical Architecture

### 1.1 Motivation and Problem Statement

The central question addressed is whether foundation models trained via sequence prediction recover genuine world models—compact representations capturing causal structure—or develop alternative computational strategies. The no-free-lunch theorem establishes that every learning algorithm exhibits inductive bias toward specific function classes when extrapolating from limited data. This observation provides the conceptual foundation: a model having learned a postulated world model should exhibit inductive bias toward functions obeying that model's state structure.

The evaluation challenge arises from the mismatch between foundation model outputs (prediction functions from data) and world model specifications (state space structure over data). Existing approaches either mechanistically probe internal representations or statically evaluate single-task performance. Neither captures how models behave when adapted to new tasks—the primary use case for foundation models.

### 1.2 Architecture Components

**State Space and World Model Formalism**

A world model consists of state space Φ and mapping ϕ: X → Φ associating inputs with states. Dataset D = {(x₁,y₁),...,(xₙ,yₙ)} is consistent with the world model when each output is a deterministic function of state: y = g(ϕ(x)) for some g: Φ → Y. Admissible functions G govern state-to-output relationships. For orbital mechanics, states correspond to 6D vectors encoding relative positions, velocities, and masses; G comprises smooth functions respecting Newtonian dynamics.

**Extrapolative Predictability**

Given foundation model m̂ producing prediction function m̂_D when applied to dataset D, sampling distribution P_D over consistent datasets, and sampling distribution P_X over inputs, extrapolative predictability between inputs xᵢ and xⱼ is defined as:

```
I^(xᵢ,xⱼ) = -min_{h∈H} E_{D∼P}[ℓ(h(m̂_D(xᵢ)), m̂_D(xⱼ))]
```

where H is a predictor family and ℓ is a loss function. Higher values indicate stronger predictability of one input's extrapolations from another's across synthetic datasets.

**Oracle Foundation Model**

The oracle foundation model, given access to true state space Φ and admissible functions G, returns:

```
m*_D = argmin_{g∈G} (1/|D|) Σ_{(xᵢ,yᵢ)∈D} ℓ(g(ϕ(xᵢ)), yᵢ)
```

Oracle extrapolative predictability I*(xᵢ,xⱼ) serves as calibration baseline.

**Inductive Bias Metric**

The foundation model's inductive bias toward the world model for oracle predictability range [s, s̄] is:

```
IB(s,s̄) = E_{Xᵢ,Xⱼ}[I^(Xᵢ,Xⱼ) | s ≤ I*(Xᵢ,Xⱼ) ≤ s̄]
```

Perfect alignment appears as a 45-degree line when plotting I^ against I*.

**Special Case: Finite State Space, Binary Outputs**

For binary Y = {0,1} and finite Φ, the framework reduces to two metrics. R-IB (Respecting State):

```
E_{Xᵢ,Xⱼ,D}[1(m̂_D(Xᵢ), m̂_D(Xⱼ)) | ϕ(Xᵢ) = ϕ(Xⱼ)]
```

D-IB (Distinguishing State):

```
1 - E_{Xᵢ,Xⱼ,D}[1(m̂_D(Xᵢ), m̂_D(Xⱼ)) | ϕ(Xᵢ) ≠ ϕ(Xⱼ)]
```

R-IB measures prediction consistency for same-state inputs; D-IB measures prediction differentiation for different-state inputs.

### 1.3 Training or Pre-Training Protocol

**Orbital Mechanics Dataset**

Training corpus consists of 10M sequences (20B tokens) simulating planetary orbits. Each sequence encodes K planets orbiting a sun across 1,000 observations at either 6-month or 1-week intervals. Initial conditions sample from empirical exoplanet distributions: eccentricity from Beta(α=0.867, β=3.03), semi-major axis from Unif(0.3, 42) AU, planet masses from LogUniform(10⁻⁷, 10⁻³) solar masses, stellar masses from Unif(0.5, 5) solar masses. Trajectories computed via Kepler's equation; planet-planet interactions omitted due to mass ratios.

Positions discretized into 7,000 bins per coordinate spanning [-50, 50] AU. Transformer (109M parameters, 12 layers, 12 attention heads, 768 embedding dimensions) trained with cross-entropy loss for 25 epochs using 8 H100 GPUs. Learning rate 6×10⁻⁴ with 2,000-step warmup, weight decay 0.1, gradient clipping at 1.

**Lattice Problems**

State space Φ = {1,2,...,S} for S ∈ {2,3,4,5}. Language Σ = {L,⊥,R} encodes agent movement on line segment. State transitions: R increments state (capped at S), L decrements (capped at 1), ⊥ maintains state. Training set contains 10M tokens from random valid sequences of length 100.

**Othello**

Game sequences tokenized as move sequences (max 60 tokens, vocabulary 60 squares). True state space corresponds to all 8×8 board configurations. Training set comprises 20M games; hold-out set 3.8M games.

**Model Architectures**

Five architectures evaluated: RNN (6 layers, 768 dimensions for Othello; 2 layers for lattice), LSTM (same specifications), Transformer (12 layers, 12 heads, 768 dimensions), Mamba (24 layers, 768 dimensions, SSM expansion factor 16, block expansion 2, convolution width 4), Mamba-2 (same specifications with Mamba-2 mixer). All models trained with Adam (learning rate 6×10⁻⁴, 2,000 warmup steps, weight decay 0.1, gradient clipping 1).

### 1.4 Performance Impact

**Next-Token Prediction Performance**

Orbital mechanics transformer achieves R² > 0.9999 on held-out trajectories, significantly exceeding baselines (previous position: MSE 1.16×10⁻⁴ at 1 step; per-orbit mean: MSE 7.53×10⁻²). For lattice problems with 5 states, all models achieve 100% legal move prediction. Othello models generate legal moves 99.2-99.9% of time across architectures.

**Inductive Bias Performance**

Orbital mechanics transformer exhibits poor alignment with Newtonian oracle (both linear and MLP oracles tested). Points cluster away from 45-degree calibration line, indicating extrapolations depend on non-Newtonian factors.

Lattice results show R-IB and D-IB degradation as state count increases:

| States | Transformer R-IB | Transformer D-IB |
|--------|------------------|------------------|
| 2 | 0.89 | 0.95 |
| 3 | 0.71 | 0.88 |
| 4 | 0.58 | 0.75 |
| 5 | 0.48 | 0.68 |

Othello results demonstrate consistently low inductive biases:

| Model | R-IB | D-IB |
|-------|------|------|
| RNN | 0.632 (±0.023) | 0.797 (±0.023) |
| LSTM | 0.563 (±0.030) | 0.610 (±0.034) |
| Transformer | 0.703 (±0.025) | 0.624 (±0.033) |
| Mamba | 0.682 (±0.021) | 0.728 (±0.027) |
| Mamba-2 | 0.653 (±0.022) | 0.694 (±0.029) |

Untrained models (random initialization) perform comparably or worse, confirming pretraining provides limited inductive bias toward true state spaces.

---

## 2. Post-Training or Optimization Methods

**Inductive Bias Probe Implementation**

For finite state spaces with binary outputs, the procedure generates 100 synthetic datasets D₁,...,D₁₀₀, each containing 100 examples. Sequences sampled uniformly; outputs assigned via Bernoulli(0.5) with state-consistency constraint (same state → same output). Foundation model fine-tuned separately on each dataset, yielding 100 prediction functions m̂(·;D₁),...,m̂(·;D₁₀₀).

Metrics computed over 2,000 randomly sampled hold-out inputs xₖ₁,...,xₖ₁₀₀. R-IB and D-IB calculated as empirical averages over same-state and different-state pairs respectively.

For continuous outputs (orbital mechanics), datasets consist of linear projections of 6D state vectors. Projection matrix selection: sample 50 random Gaussian (6×1) matrices, select matrix maximizing Spearman correlation between pairwise Euclidean distances in 6D and projected 1D spaces. This procedure ensures maximal information preservation under dimensionality reduction.

Fine-tuning uses 100 examples per dataset across 100 synthetic datasets. Extrapolative predictability computed via pairwise Euclidean distances, binned into 20 equal-width ranges for calibration curve construction.

**Force Vector Prediction**

Force vector prediction experiments use sequence-to-sequence datasets where inputs are trajectories and outputs are force magnitudes F at each timestep. Newton's law specifies F = G(m₁m₂/r²), where r is planet-sun distance and m₁,m₂ are masses.

Fine-tuning procedure: transformer pretrained on orbital trajectories adapted to predict force vectors using 1% of true forces as training data (9,000 two-body systems). Training runs 10,000 steps (batch size 64) with learning rate 2×10⁻⁴ (grid search range 10⁻⁶ to 5×10⁻⁴). Checkpoint selected based on lowest held-out MSE.

Extrapolation restricted to 5,000 timesteps with states most similar to training set (measured via Euclidean distance), ensuring evaluation focuses on interpolation rather than extreme extrapolation.

**Symbolic Regression**

PySR library performs symbolic regression on predicted force magnitudes. Search space constrained to expressions with maximum size 20, binary operators {addition, multiplication}, unary operators {sine, cosine, exponentiation, inverse}. Loss function applies zero penalty within 10⁻⁸ tolerance, otherwise penalizes absolute distance.

Five independent runs sample different 1,000-point subsets from test set (representing different "galaxies"). Each run executes 100 iterations with three random restarts.

**LLM Experiments**

Three reasoning models (o3, Claude Sonnet 4, Gemini 2.5 Pro) evaluated via in-context learning. Five random solar systems generated (450 observations each). Prompt describes data structure and provides 10 randomly selected true force magnitudes (2.2% of data). Models instructed to predict remaining force values without explicit indication that outputs represent physical forces.

---

## 3. Agentic or System-Level Design (if applicable)

Not applicable per source document.

---

## 4. Benchmark Performance and Ablations

**Force Law Recovery Results**

Symbolic regression on transformer predictions yields nonsensical equations varying by galaxy sample:

| Galaxy | Recovered Force Law |
|--------|---------------------|
| 1 | F ∝ sin(sin(1/sin(r-0.24))+1.45)×(1/(1/r)+m₂) |
| 2 | F ∝ cos(cos(2.19×m₁)) |
| 3 | F ∝ cos(sin(0.48/m₁)) |
| 4 | F ∝ sin(r+8569.2+1/m₁) |
| 5 | F ∝ cos(cos(e^(m₂))) |

Ground truth: F ∝ m₁m₂/r²

Oracle model (k-nearest neighbors with k=2 on true state) recovers correct gravitational law for all five galaxy samples, demonstrating procedure feasibility when extrapolating from proper world model.

**LLM Force Prediction**

Large language models exhibit poor force magnitude prediction despite training on physics-rich text corpora:

| Model | Recovered Force Law |
|-------|---------------------|
| o3 | F ∝ m₁ |
| Claude Sonnet 4 | F ∝ 1/m₂⁻⁰·⁵⁰ |
| Gemini 2.5 Pro | F ∝ m₁ |

Predictions systematically deviate from true magnitudes across timesteps.

**Transfer Learning Performance**

Othello models fine-tuned to predict three board-derived functions: Majority Tiles (more black/white pieces), Board Balance (top-half vs bottom-half black piece count), Edge Balance (edge vs center black piece count). Correlation between inductive bias ratio (R-IB/(1-D-IB)) and transfer performance:

| Task | NLL Correlation | ACC Correlation |
|------|-----------------|-----------------|
| Majority Tiles | 0.462 | 0.477 |
| Board Balance | 0.610 | 0.653 |
| Edge Balance | 0.970 | 0.960 |

Pretrained transformer with highest R-IB achieves best transfer: Majority Tiles NLL 0.100 (untrained: 0.497), Board Balance NLL 0.086 (untrained: 0.340), Edge Balance NLL 0.013 (untrained: 0.075).

**Ablation Studies**

Fine-tuning iteration count ablation (Othello, 100 examples):

| Iterations | LSTM R-IB | LSTM D-IB | Transformer R-IB | Transformer D-IB |
|-----------|-----------|-----------|------------------|------------------|
| 10 | 0.805 | 0.510 | 0.775 | 0.585 |
| 50 | 0.576 | 0.605 | 0.712 | 0.619 |
| 100 | 0.563 | 0.610 | 0.703 | 0.624 |
| 500 | 0.553 | 0.615 | 0.714 | 0.629 |

Fine-tuning example count ablation (Othello, 100 iterations):

| Examples | LSTM R-IB | LSTM D-IB | Transformer R-IB | Transformer D-IB |
|----------|-----------|-----------|------------------|------------------|
| 10 | 0.750 | 0.374 | 0.862 | 0.363 |
| 50 | 0.625 | 0.543 | 0.721 | 0.610 |
| 100 | 0.563 | 0.610 | 0.703 | 0.624 |
| 500 | 0.483 | 0.832 | 0.578 | 0.853 |

Results show limited sensitivity to hyperparameters; poor inductive bias persists across configurations.

**Next-Token Partition Analysis**

Coarsened state space q groups states by legal next-token sets: q(x) = q(x') iff NextTokens(ϕ(x)) = NextTokens(ϕ(x')). Decomposition of D-IB:

D-IB_q= : Predictability for different states with same legal tokens
D-IB_q≠ : Predictability for different states with different legal tokens

Results demonstrate D-IB_q= < D-IB_q≠ across all models:

| Model | Lattice D-IB_q= | Lattice D-IB_q≠ | Othello D-IB_q= | Othello D-IB_q≠ |
|-------|-----------------|-----------------|-----------------|-----------------|
| RNN | 0.740 (±0.042) | 0.844 (±0.034) | 0.521 (±0.031) | 0.798 (±0.023) |
| LSTM | 0.873 (±0.051) | 0.952 (±0.034) | 0.519 (±0.035) | 0.610 (±0.034) |
| Transformer | 0.626 (±0.037) | 0.710 (±0.037) | 0.458 (±0.033) | 0.625 (±0.033) |
| Mamba | 0.764 (±0.040) | 0.933 (±0.035) | 0.485 (±0.030) | 0.729 (±0.027) |
| Mamba-2 | 0.778 (±0.042) | 0.920 (±0.033) | 0.553 (±0.032) | 0.694 (±0.029) |

Gap statistically significant for all architectures, indicating models group distinct states by legal next-token partitions rather than true state structure.

**Board Reconstruction Experiment**

Mamba model pretrained on next-token prediction fine-tuned to predict full Othello boards from sequences. Despite imperfect board reconstruction, predicted boards frequently preserve legal move sets: when predicted board differs from true board, 73% still yield identical legal next-move sets. Models recover sufficient board information to determine legal tokens without capturing complete state.

---

## 5. Key Technical Takeaways

- Foundation models achieving >99% next-token accuracy on sequence prediction tasks do not necessarily develop inductive biases toward underlying world models governing those sequences
- Transformers trained on 10M orbital trajectories with R² > 0.9999 position prediction accuracy recover nonsensical force laws fundamentally inconsistent with Newtonian mechanics (e.g., trigonometric functions of mass/distance rather than inverse-square relationships)
- Symbolic regression on force predictions yields different equations for different data subsets, suggesting piecemeal heuristics rather than unified physical laws
- Inductive bias probe metrics (R-IB, D-IB) demonstrate systematic degradation as state space complexity increases, with transformer R-IB dropping from 0.89 (2 states) to 0.48 (5 states) in lattice problems
- Models develop coarsened representations based on legal next-token partitions: D-IB_q= consistently lower than D-IB_q≠ across architectures (e.g., transformer Othello: 0.458 vs 0.625), indicating grouping of distinct states sharing identical legal move sets
- Transfer learning performance correlates strongly with inductive bias metrics (r = 0.96-0.97 for edge balance task), validating probe's predictive utility for downstream task adaptation
- Recurrent architectures (RNN, LSTM, Mamba) generally outperform transformers on inductive bias metrics despite comparable next-token accuracy
- Large language models (o3, Claude Sonnet 4, Gemini 2.5 Pro) exhibit similar failures when evaluated via in-context learning on orbital mechanics, recovering mass-only or power-law relationships rather than inverse-square gravitational law
- Board reconstruction experiments reveal models recover partial state sufficient for legal move determination without capturing complete board configuration
- Oracle models with direct state access consistently recover correct physical laws across all experimental conditions, demonstrating methodology validity

---

## 6. Conclusion

The inductive bias probe framework reveals a systematic dissociation between sequence prediction accuracy and world model acquisition in foundation models. Models achieving near-perfect next-token prediction across orbital mechanics, lattice problems, and board games fail to develop inductive biases aligned with underlying state spaces. Instead, analysis indicates reliance on task-specific heuristics and coarsened representations based on legal next-token partitions. The framework provides empirical methodology for differentiating superficial pattern matching from genuine causal understanding, with implications for foundation model deployment in scientific domains requiring robust physical reasoning. The strong correlation between inductive bias metrics and transfer learning performance (r > 0.96) validates the probe's utility for predicting model behavior on novel downstream tasks. Future work should prioritize automated discovery of implicit world models from foundation model behavior and development of training procedures encouraging alignment with target world models rather than next-token optimization alone.

---

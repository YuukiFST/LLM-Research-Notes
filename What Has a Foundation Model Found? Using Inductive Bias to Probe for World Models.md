# Inductive Bias Probes for Foundation Model World Models

## Executive Summary

This paper introduces inductive bias probes, a behavioral testing methodology that evaluates whether foundation models develop internal world models during sequence prediction training. Unlike mechanistic interpretability approaches that examine internal representations, inductive bias probes measure how models extrapolate when adapted to small synthetic datasets derived from a postulated world model. Empirical evaluation across three domains—orbital mechanics, lattice problems, and Othello—reveals that models achieving >99% next-token prediction accuracy systematically fail to develop inductive biases aligned with underlying state structures. In orbital mechanics, transformers trained on planetary trajectories recover nonsensical force laws (F ∝ sin(sin^(-1)(r-0.24))+1.45 × 1/(1/r)+m₂) rather than Newton's gravitational law (F ∝ m₁m₂/r²), with different incoherent laws recovered across different galaxy samples. Analysis indicates models develop task-specific heuristics based on legal next-token partitions rather than coherent world models, evidenced by significantly lower D-IB scores for distinct states sharing legal moves (D-IBq= < D-IBq≠ across all architectures).

---

## 1. Technical Architecture

### 1.1 Motivation and Problem Statement

The central hypothesis is that sequence prediction can uncover deeper domain structure analogous to how Kepler's planetary motion predictions preceded Newton's mechanics. The evaluation challenge is that foundation models and world models operate in different spaces: models produce predictive functions from data, while world models describe implicit state structures. Standard approaches fail: mechanistic probing of weight-level representations faces interpretability challenges and may not reflect actual behavior on new data; static behavioral evaluation on single tasks does not capture how foundation models are deployed for adaptation.

### 1.2 Architecture Components

**World Model Formalization**

A world model consists of:
- State space Φ
- Mapping ϕ: X → Φ associating inputs with states
- Admissible function set G where g ∈ G: Φ → Y
- Dataset D = {(x₁,y₁),...,(xₙ,yₙ)} is consistent if ∀(x,y) ∈ D: y = g(ϕ(x)) for some g ∈ G

**Foundation Model Definition**

A foundation model is a learning algorithm that maps dataset D to prediction function:

```
m̂_D: X → Y
```

Examples include pre-trained models fine-tuned on D or LLMs with D provided in-context.

**Inductive Bias Probe Framework**

The probe leverages the no-free-lunch theorem: no learning algorithm dominates on average across all possible functions. Each algorithm has inductive bias toward functions it tends to learn from limited data. The probe tests whether this bias aligns with the world model's state structure.

**Binary Output, Finite State Space (Special Case)**

For Y = {0,1} and finite Φ, two metrics are defined:

Respect-state Inductive Bias (R-IB):
```
E[1(m̂_D(Xᵢ), m̂_D(Xⱼ)) | ϕ(Xᵢ) = ϕ(Xⱼ)]
```

Distinguish-state Inductive Bias (D-IB):
```
1 - E[1(m̂_D(Xᵢ), m̂_D(Xⱼ)) | ϕ(Xᵢ) ≠ ϕ(Xⱼ)]
```

where expectations are over (Xᵢ,Xⱼ) ∼ P_X × P_X and D ∼ P_D.

**General Case: Extrapolative Predictability**

For continuous outputs and general state spaces, extrapolative predictability between inputs xᵢ, xⱼ is:

```
Î(xᵢ, xⱼ) = -min_{h∈H} E_{D∼P}[ℓ(h(m̂_D(xᵢ)), m̂_D(xⱼ))]
```

where H is a predictor family and ℓ: Y × Y → R⁺ is a loss function.

Oracle extrapolative predictability for a model with access to true state:

```
I*(xᵢ, xⱼ) = -min_{h∈H} E_{D∼P}[ℓ(h(m*_D(xᵢ)), m*_D(xⱼ))]

m*_D = argmin_{g∈G} (1/|D|) Σ_{(xᵢ,yᵢ)∈D} ℓ(g(ϕ(xᵢ)), yᵢ)
```

Inductive bias metric (calibration curve):

```
IB(s, s̄) = E_{Xᵢ,Xⱼ}[Î(Xᵢ, Xⱼ) | s ≤ I*(Xᵢ, Xⱼ) ≤ s̄]
```

computed over grid 0 = s₀ < s₁ < ... < sₘ. Perfect alignment with oracle yields 45-degree line.

### 1.3 Training or Pre-Training Protocol

**Orbital Mechanics**

- Dataset: 10M sequences, 20B tokens
- Sequence structure: 1000 observations per solar system, interleaved planet coordinates
- Sampling: K planets per system (K ∼ Unif[1,10]), eccentricity ∼ Beta(α=0.867, β=3.03), semi-major axis ∼ Unif(0.3, 42) AU, planet mass ∼ LogUniform(10⁻⁷, 10⁻³), star mass ∼ Unif(0.5, 5)
- Time intervals: 50% use 6-month steps, 50% use 1-week steps (indicated by special token)
- Coordinates: Discretized into 7K bins per dimension over [-50, 50] AU
- Architecture: 109M parameter transformer (12 layers, 12 heads, 768 dimensions)
- Training: 25 epochs, 8 H100 GPUs, cross-entropy loss on discretized coordinates

**Lattice Problems**

- State space: Φ = {1, 2, ..., S} for S ∈ {2,3,4,5}
- Vocabulary: Σ = {L, ⊥, R} (left, stay, right)
- Dynamics: R increases state by 1, L decreases by 1, ⊥ maintains state; boundaries prevent L at state 1, R at state S
- Training set: 10M tokens, sequences of length 100
- Holdout: 100K tokens

**Othello**

- State space: All valid 8×8 board configurations
- Sequence encoding: Up to 60 moves, each token indicates square placement (60 possible positions)
- Training set: 20M games
- Holdout: 3.8M games

**Model Architectures** (all domains):

- RNN: 6 layers (2 for lattice), 768 dimensions
- LSTM: Same as RNN with LSTM layers
- Transformer: 12 layers, 12 heads, 768 dimensions
- Mamba: 24 layers, 768 dimensions, SSM expansion 16, block expansion 2, conv width 4
- Mamba-2: Same as Mamba with Mamba-2 mixer

Training: Adam optimizer, learning rate 6e-4, 2000 warmup steps, weight decay 0.1, gradient clipping at 1.0

### 1.4 Performance Impact

**Next-Token Prediction Accuracy**

All models achieve near-perfect legal token prediction:

| Domain | Lattice (5 states) | Othello |
|--------|-------------------|---------|
| RNN | 1.000 | 0.992 |
| LSTM | 1.000 | 0.996 |
| Transformer | 1.000 | 0.999 |
| Mamba | 1.000 | 0.999 |
| Mamba-2 | 1.000 | 0.999 |

Orbital mechanics: Transformer achieves R² > 0.9999 on held-out trajectories, significantly outperforming baselines (per-orbit mean MSE: 7.53×10⁻², previous position MSE: 1.16×10⁻⁴, transformer MSE: 1.90×10⁻⁸ for 1-step prediction).

**Inductive Bias Degradation**

Despite prediction accuracy, inductive bias metrics reveal systematic failures. For Othello:

| Model | R-IB (pretrained) | D-IB (pretrained) | R-IB (untrained) | D-IB (untrained) |
|-------|------------------|------------------|-----------------|-----------------|
| RNN | 0.632 ± 0.023 | 0.797 ± 0.023 | 0.228 ± 0.016 | 0.990 ± 0.002 |
| LSTM | 0.563 ± 0.030 | 0.610 ± 0.034 | 0.438 ± 0.030 | 0.681 ± 0.031 |
| Transformer | 0.703 ± 0.025 | 0.624 ± 0.033 | 0.708 ± 0.022 | 0.843 ± 0.021 |
| Mamba | 0.682 ± 0.021 | 0.728 ± 0.027 | 0.303 ± 0.016 | 0.929 ± 0.009 |
| Mamba-2 | 0.653 ± 0.022 | 0.694 ± 0.029 | 0.468 ± 0.019 | 0.896 ± 0.016 |

Lattice results show degradation as state complexity increases. For 5-state lattice:

| Model | R-IB | D-IB |
|-------|------|------|
| RNN | 0.574 ± 0.026 | 0.803 ± 0.032 |
| LSTM | 0.782 ± 0.021 | 0.921 ± 0.030 |
| Transformer | 0.483 ± 0.031 | 0.677 ± 0.034 |
| Mamba | 0.571 ± 0.023 | 0.866 ± 0.029 |
| Mamba-2 | 0.617 ± 0.021 | 0.864 ± 0.029 |

**Transfer Learning Correlation**

Strong correlation (0.462-0.970) between IB metrics (R-IB/D-IB ratio) and transfer performance on state-dependent tasks:

| Task | Transformer NLL | Mamba-2 NLL | Baseline NLL |
|------|----------------|-------------|--------------|
| Majority Tiles | 0.100 ± 0.002 | 0.069 ± 0.002 | 0.497 ± 0.004 |
| Board Balance | 0.086 ± 0.002 | 0.059 ± 0.002 | 0.340 ± 0.005 |
| Edge Balance | 0.013 ± 0.001 | 0.012 ± 0.002 | 0.075 ± 0.002 |

---

## 2. Post-Training or Optimization Methods

**Synthetic Dataset Generation**

For binary outputs and finite state spaces, datasets are generated by:
1. Randomly assigning each unique state ϕ(x) to output 0 or 1 with probability 0.5
2. Ensuring consistency: all inputs mapping to same state receive identical outputs
3. Sampling 100 examples per dataset
4. Generating 100 independent datasets

For continuous outputs (orbital mechanics):
1. Sample 100 sequences from training distribution
2. Extract Newtonian state vectors (masses, relative positions, relative velocities)
3. Generate 50 random 6×1 Gaussian projection matrices
4. Select projection maximizing Spearman correlation between 6D Euclidean distances and 1D projected distances
5. Sample one projected point per sequence
6. Repeat to create 100 datasets of 100 examples each

**Fine-Tuning Protocol**

Each model is independently fine-tuned on each synthetic dataset:
- Learning rates: Grid search from 1e-6 to 5e-4 (optimal: 2e-4 for orbital mechanics)
- Iterations: 100 (ablated: {10, 50, 100, 500})
- Training examples: 100 (ablated: {10, 50, 100, 500})
- Checkpoint selection: Lowest validation loss

**Force Prediction Task**

Specific procedure for testing Newtonian mechanics:
1. Create sequence-to-sequence dataset mapping trajectories to force vectors F at each timestep
2. Fine-tune on 1% of solar system force observations (training: 9K two-body systems, test: 1K systems)
3. Add 2 random timesteps per test sequence to training set to enable partial observation extrapolation
4. Select 5,000 test timesteps with states most similar to training (via Euclidean distance)
5. Impute force magnitude predictions
6. Apply symbolic regression (PySR) with constraints: max size 20, binary operators {+, ×}, unary operators {sin, cos, exp, inverse}
7. Loss function: 0 penalty within 1e-8, absolute distance otherwise
8. Run 3 random restarts of 100 iterations each, select best score

**Symbolic Regression**

Automated discovery of functional forms using PySR library applied to model predictions to recover implicit physical laws.

---

## 3. Agentic or System-Level Design (if applicable)

Not applicable per source document.

---

## 4. Benchmark Performance and Ablations

**Orbital Mechanics: Trajectory Prediction**

| Model | 1 step MSE | 5 steps MSE | 100 steps MSE |
|-------|-----------|-------------|---------------|
| Per-orbit mean | (7.53±0.59)×10⁻² | (5.53±0.58)×10⁻² | (1.39±0.08)×10⁻¹ |
| Previous position | (1.16±0.21)×10⁻⁴ | (1.37±0.38)×10⁻⁴ | (4.04±0.47)×10⁻² |
| Transformer | (1.90±0.45)×10⁻⁸ | (1.56±0.45)×10⁻⁸ | (3.74±3.37)×10⁻⁵ |

**Force Law Recovery via Symbolic Regression**

Ground truth: F ∝ m₁m₂/r²

Transformer-recovered laws across 5 galaxy samples:
- Galaxy 1: F ∝ [sin(1/sin(r-0.24))+1.45] × 1/(1/r+m₂)
- Galaxy 2: F ∝ cos[cos(2.19×m₁)]
- Galaxy 3: F ∝ cos[sin(0.48/m₁)]
- Galaxy 4: F ∝ sin(r+8569.2+1/m₁)
- Galaxy 5: F ∝ cos[cos(e^m₂)]

Oracle (k-NN on true state): Correctly recovers F ∝ m₁m₂/r² for all 5 samples.

**LLM Force Prediction**

| Model | Recovered Law |
|-------|--------------|
| o3 | F ∝ m₁ |
| Claude Sonnet 4 | F ∝ 1/m₂⁻⁰·⁵⁰ |
| Gemini 2.5 Pro | F ∝ m₁ |

All LLMs provided 450 observations with 10 labeled force magnitudes in-context, failing to extrapolate Newtonian mechanics.

**Fine-Tuning Iterations Ablation (Othello)**

| Model | 10 iter R-IB | 10 iter D-IB | 100 iter R-IB | 100 iter D-IB | 500 iter R-IB | 500 iter D-IB |
|-------|-------------|-------------|--------------|--------------|--------------|--------------|
| Transformer | 0.775±0.022 | 0.585±0.032 | 0.703±0.025 | 0.624±0.033 | 0.714±0.024 | 0.629±0.033 |
| Mamba | 0.775±0.019 | 0.730±0.025 | 0.682±0.021 | 0.728±0.027 | 0.683±0.021 | 0.710±0.028 |

**Training Examples Ablation (Othello)**

| Model | 10 ex R-IB | 10 ex D-IB | 100 ex R-IB | 100 ex D-IB | 500 ex R-IB | 500 ex D-IB |
|-------|-----------|-----------|------------|------------|------------|------------|
| Transformer | 0.862±0.019 | 0.363±0.038 | 0.703±0.025 | 0.624±0.033 | 0.578±0.021 | 0.853±0.018 |
| Mamba | 0.821±0.020 | 0.456±0.039 | 0.682±0.021 | 0.728±0.027 | 0.654±0.018 | 0.864±0.014 |

**Next-Token Partition Analysis**

Decomposition of D-IB into D-IBq= (different states, same legal moves) and D-IBq≠ (different states, different legal moves):

Lattice (5 states):

| Model | D-IBq= | D-IBq≠ |
|-------|--------|--------|
| RNN | 0.740±0.042 | 0.844±0.034 |
| LSTM | 0.873±0.051 | 0.952±0.034 |
| Transformer | 0.626±0.037 | 0.710±0.037 |
| Mamba | 0.764±0.040 | 0.933±0.035 |
| Mamba-2 | 0.778±0.042 | 0.920±0.033 |

Othello:

| Model | D-IBq= | D-IBq≠ |
|-------|--------|--------|
| RNN | 0.521±0.031 | 0.798±0.023 |
| LSTM | 0.519±0.035 | 0.610±0.034 |
| Transformer | 0.458±0.033 | 0.625±0.033 |
| Mamba | 0.485±0.030 | 0.729±0.027 |
| Mamba-2 | 0.553±0.032 | 0.694±0.029 |

All models show D-IBq= < D-IBq≠ (statistically significant), indicating inductive bias toward legal next-token partitions rather than full state.

**Modified Two-Body Setup**

Transformer trained on 10M two-body systems (equal masses, center-of-mass frame):
- Inductive bias probe: Similar poor performance to multi-planet setup
- Force law recovery: F ∝ m₁ × 1/exp(r) (nonsensical)
- Modified evaluation (partial trajectory observation): No improvement

---

## 5. Key Technical Takeaways

- Foundation models achieve near-perfect next-token prediction (>99% legal moves) without developing inductive biases toward underlying world models, as measured by behavioral extrapolation on synthetic adaptation tasks
- Transformers trained on 20B tokens of orbital trajectories fail to recover Newtonian mechanics, producing different nonsensical force laws for different data samples rather than a single coherent physical theory
- R-IB and D-IB metrics function analogously to precision-recall: high R-IB alone (same predictions for same states) is trivially achievable by constant prediction; high D-IB alone (different predictions for different states) requires distinguishing state structure
- Inductive bias probe methodology is representation-invariant, depending only on state equality rather than mechanistic encoding, addressing limitations of standard linear probes which vary with equivalent state representations
- All evaluated architectures (RNN, LSTM, Transformer, Mamba, Mamba-2) exhibit statistically significant preference for next-token partitions (D-IBq= < D-IBq≠), grouping distinct states with identical legal moves rather than respecting full state structure
- Transfer learning performance on state-dependent functions correlates strongly (r=0.462-0.970) with inductive bias metrics, indicating practical utility beyond theoretical measurement
- Pretraining provides marginal improvement over random initialization for some architectures but does not fundamentally alter inductive bias alignment with world models
- Fine-tuning iteration count (10-500) and training example count (10-500) ablations show minimal effect on inductive bias metrics, suggesting fundamental architectural limitations rather than optimization issues
- Board reconstruction experiments in Othello reveal models recover sufficient partial state to predict legal moves without reconstructing full board state, evidencing coarsened internal representations
- LLMs (o3, Claude Sonnet 4, Gemini 2.5 Pro) with extensive physics knowledge in training corpora fail force prediction tasks when provided partial trajectory data in-context, recovering overly simplistic laws (F ∝ m₁)

---

## 6. Conclusion

This work establishes inductive bias probes as a behavioral evaluation methodology for assessing whether foundation models develop world models during sequence prediction training. The method addresses fundamental limitations of mechanistic interpretability by directly measuring extrapolation behavior across synthetic adaptation tasks rather than analyzing internal representations. Empirical findings demonstrate systematic failure: models achieving near-perfect predictive performance on next-token objectives develop inductive biases toward task-specific heuristics (particularly legal next-token partitions) rather than coherent state structures. In orbital mechanics, this manifests as recovery of different nonsensical physical laws across data samples rather than universal Newtonian mechanics. The strong correlation between inductive bias metrics and transfer learning performance establishes practical utility for predicting model behavior on downstream state-dependent tasks. The framework requires specification of a candidate world model, limiting applicability for discovering unknown implicit representations—future work should prioritize automated construction of world models from observed extrapolation patterns. The findings challenge the foundational premise that sequence prediction inherently uncovers deeper domain understanding, suggesting architectural innovations or training objectives explicitly incorporating inductive biases toward state structure may be necessary for foundation models to make the leap from pattern matching to genuine world modeling.

---

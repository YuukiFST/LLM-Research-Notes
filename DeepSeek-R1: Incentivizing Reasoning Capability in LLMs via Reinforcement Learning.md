# DeepSeek-R1: Scaling Reasoning Through Reinforcement Learning

## Executive Summary

DeepSeek-R1 represents a paradigm shift in developing reasoning capabilities for Large Language Models (LLMs). Unlike traditional approaches that rely heavily on supervised fine-tuning (SFT), DeepSeek demonstrates that **pure reinforcement learning (RL) can autonomously develop sophisticated reasoning behaviors**—including self-verification, reflection, and extended chain-of-thought (CoT) generation—without any supervised cold-start data.

The paper introduces two model families:

1. **DeepSeek-R1-Zero**: A pure RL-trained model achieving 71.0% pass@1 on AIME 2024 (matching OpenAI-o1-0912 performance)
2. **DeepSeek-R1**: A production model incorporating minimal cold-start data, achieving 79.8% on AIME 2024 (surpassing OpenAI-o1-1217)

Additionally, DeepSeek releases six distilled dense models (1.5B-70B parameters) that demonstrate reasoning capabilities can be effectively transferred to smaller models through knowledge distillation, with the 7B model outperforming GPT-4o on mathematical benchmarks.

**Key Innovation**: The research validates that reasoning emerges naturally from RL optimization with outcome-based rewards alone, without requiring process supervision, Monte Carlo Tree Search (MCTS), or process reward models (PRMs).

---

## 1. Methodology & Architecture

### 1.1 Core Training Framework: Group Relative Policy Optimization (GRPO)

DeepSeek-R1 employs **GRPO** (Shao et al., 2024), a cost-efficient RL algorithm that eliminates the need for a separate critic model. The approach:

- Samples groups of outputs {o₁, o₂, ..., oG} from the policy
- Computes advantages using group-level statistics rather than value function estimates
- Optimizes the policy objective:

```
J_GRPO(θ) = E[min(π_θ(o|q)/π_old(o|q) * A_i, clip(...)) - β*D_KL(π_θ||π_ref)]
```

Where advantages are normalized within each group:
```
A_i = (r_i - mean(rewards)) / std(rewards)
```

**Cost Benefit**: By removing the critic model (typically the same size as the policy), GRPO reduces RL training costs by approximately 50% compared to PPO-style algorithms.

### 1.2 DeepSeek-R1-Zero: Pure RL from Base Model

**Training Protocol**:
- **Base Model**: DeepSeek-V3-Base (671B total parameters, 37B activated per forward pass)
- **No SFT pretraining**: RL applied directly to the base model
- **Template Design**: Simple structural template requiring `<think>` reasoning followed by `<answer>` tags
- **Reward System**: Rule-based only (no neural reward models)
  - **Accuracy rewards**: Deterministic verification for math/code problems
  - **Format rewards**: Enforces correct tag structure

**Key Finding**: The model naturally developed sophisticated behaviors through RL alone:
- Self-verification mechanisms
- Reflective reasoning patterns
- Extended thinking time allocation (thousands of tokens)
- "Aha moments" where the model spontaneously re-evaluates approaches

**Performance Trajectory**:
- Initial AIME 2024 pass@1: 15.6%
- Final AIME 2024 pass@1: 71.0%
- With consensus@64 (majority voting): 86.7%

### 1.3 DeepSeek-R1: Multi-Stage Training Pipeline

The production model incorporates four training stages:

#### Stage 1: Cold Start Data Collection
- **Data Volume**: Thousands of long CoT examples (exact count not disclosed)
- **Sources**: Few-shot prompting, direct prompting with reflection, DeepSeek-R1-Zero outputs, human post-processing
- **Output Format**: `|special_token|<reasoning_process>|special_token|<summary>`
- **Design Principles**: Prioritize readability, include summaries, filter language mixing

#### Stage 2: Reasoning-Oriented RL
- **Focus**: Math, coding, science, and logic reasoning tasks
- **Additional Reward**: Language consistency reward (proportion of target language tokens)
- **Trade-off**: Slight performance degradation for improved human readability
- **Final Reward**: `reward_total = reward_accuracy + reward_language_consistency`

#### Stage 3: Rejection Sampling & SFT
- **Reasoning Data**: ~600K samples via rejection sampling from RL checkpoint
  - Uses rule-based rewards for deterministic tasks
  - Applies generative reward models (DeepSeek-V3-based) for non-deterministic tasks
  - Filters chaotic outputs (language mixing, long paragraphs, excessive code blocks)
- **Non-Reasoning Data**: ~200K samples from DeepSeek-V3 pipeline
  - Domains: Writing, factual QA, self-cognition, translation
  - CoT generated for complex queries only (not for simple interactions like "hello")
- **Total Dataset**: ~800K samples
- **Training**: 2 epochs on DeepSeek-V3-Base

#### Stage 4: Comprehensive RL
- **Objective**: Align with helpfulness, harmlessness, and reasoning excellence
- **Reward Sources**:
  - Rule-based rewards for reasoning tasks
  - Neural reward models for general tasks
- **Evaluation Strategy**:
  - Helpfulness: Evaluated on summary only (minimizes CoT interference)
  - Harmlessness: Evaluated on full response (reasoning + summary)

---

## 2. Experimental Results

### 2.1 DeepSeek-R1 Performance

**Mathematical Reasoning**:
- AIME 2024: **79.8%** pass@1 (vs. OpenAI-o1-1217: 79.2%)
- MATH-500: **97.3%** pass@1 (vs. OpenAI-o1-1217: 96.4%)
- CNMO 2024: **78.8%** pass@1 (Chinese Math Olympiad)

**Coding**:
- Codeforces Rating: **2029** Elo (96.3rd percentile) - expert level
- LiveCodeBench: **65.9%** pass@1 (vs. o1-1217: 63.4%)
- SWE-Bench Verified: **49.2%** resolved (comparable to o1-1217: 48.9%)

**Knowledge Benchmarks**:
- MMLU: **90.8%** (vs. o1-1217: 91.8%)
- MMLU-Pro: **84.0%** (vs. o1-1217: reported but not specified)
- GPQA Diamond: **71.5%** (vs. o1-1217: 75.7%)

**General Capabilities**:
- AlpacaEval 2.0 LC-winrate: **87.6%**
- Arena-Hard: **92.3%** win rate

**Key Observation**: DeepSeek-R1 achieves parity with or exceeds OpenAI-o1-1217 on mathematical and coding reasoning tasks while using summary-only evaluation for general tasks to avoid length bias.

### 2.2 Distilled Model Performance

The distillation experiments reveal crucial insights:

| Model | AIME 2024 | MATH-500 | GPQA Diamond | LiveCodeBench | Codeforces Rating |
|-------|-----------|----------|--------------|---------------|-------------------|
| **GPT-4o-0513** | 9.3% | 74.6% | 49.9% | 32.9% | 759 |
| **o1-mini** | 63.6% | 90.0% | 60.0% | 53.8% | 1820 |
| **QwQ-32B-Preview** | 50.0% | 90.6% | 54.5% | 41.9% | 1316 |
| **DeepSeek-R1-Distill-Qwen-1.5B** | 28.9% | 83.9% | 33.8% | 16.9% | 954 |
| **DeepSeek-R1-Distill-Qwen-7B** | 55.5% | 92.8% | 49.1% | 37.6% | 1189 |
| **DeepSeek-R1-Distill-Qwen-14B** | 69.7% | 93.9% | 59.1% | 53.1% | 1481 |
| **DeepSeek-R1-Distill-Qwen-32B** | 72.6% | 94.3% | 62.1% | 57.2% | 1691 |
| **DeepSeek-R1-Distill-Llama-70B** | 70.0% | 94.5% | 65.2% | 57.5% | 1633 |

**Critical Finding**: The 7B distilled model (55.5% AIME) outperforms QwQ-32B-Preview (50.0%), demonstrating that **reasoning patterns from larger models transfer more effectively than reasoning patterns discovered through RL on smaller models**.

### 2.3 Distillation vs. Direct RL (Section 4.1)

An ablation comparing three approaches on Qwen-32B-Base:

1. **QwQ-32B-Preview**: Existing open-source reasoning model
2. **DeepSeek-R1-Zero-Qwen-32B**: 10K+ RL steps on 32B base model
3. **DeepSeek-R1-Distill-Qwen-32B**: SFT on DeepSeek-R1 outputs

Results show distillation significantly outperforms direct RL:
- AIME 2024: Distillation (72.6%) vs. Direct RL (47.0%)
- MATH-500: Distillation (94.3%) vs. Direct RL (91.6%)

**Implication**: For smaller models, distillation from stronger teachers is more cost-effective than large-scale RL training. However, advancing the frontier of reasoning likely still requires larger base models and extensive RL.

---

## 3. Emergent Behaviors & The "Aha Moment"

### 3.1 Self-Evolution Through RL

Figure 3 in the paper demonstrates **spontaneous test-time compute scaling**: DeepSeek-R1-Zero naturally learned to allocate more thinking time (hundreds to thousands of tokens) without explicit instruction.

**Emergent Behaviors**:
- **Reflection**: Model revisits and re-evaluates previous reasoning steps
- **Alternative exploration**: Tries multiple solution approaches
- **Extended deliberation**: Allocates more compute to harder problems

### 3.2 The "Aha Moment" Phenomenon

Table 3 showcases a remarkable behavior from an intermediate checkpoint:

```
Wait, wait. Wait. That's an aha moment I can flag here.
Let's reevaluate this step-by-step to identify if the correct sum can be...
```

The model spontaneously:
1. Recognized its initial approach might be suboptimal
2. Used anthropomorphic language ("aha moment")
3. Initiated self-correction without explicit prompting

**Significance**: This demonstrates RL's power to develop sophisticated meta-cognitive strategies purely from outcome-based rewards—the model was never taught *how* to reflect, only incentivized to produce correct answers.

---

## 4. Technical Design Decisions

### 4.1 Why Rule-Based Rewards?

DeepSeek deliberately avoided neural reward models (outcome or process-based) due to:

1. **Reward Hacking**: Neural models are vulnerable to exploitation during large-scale RL
2. **Retraining Overhead**: Updating reward models adds complexity and computational cost
3. **Simplicity**: Rule-based verification is deterministic for math/code tasks

This contrasts with approaches like OpenAI's process supervision (Lightman et al., 2023).

### 4.2 Evaluation Strategy: pass@k vs. Greedy Decoding

**Problem**: Greedy decoding on long-output reasoning models causes high repetition rates and checkpoint instability.

**Solution**: Use temperature sampling (T=0.6, top-p=0.95) with k samples:
```
pass@1 = (1/k) * Σ p_i
```

This provides more stable performance estimates across checkpoints.

### 4.3 Template Minimalism

The DeepSeek-R1-Zero template (Table 1) intentionally avoided content-specific biases:
- ❌ No mandates for reflective reasoning
- ❌ No promotion of specific problem-solving strategies
- ✅ Only structural format: `<think>` reasoning `</think>` `<answer>` answer `</answer>`

This design enables observing the model's natural progression during RL without confounding human priors.

---

## 5. Limitations & Failed Approaches

### 5.1 Documented Challenges

**Language Mixing**:
- DeepSeek-R1-Zero frequently mixed languages in CoT
- Mitigation: Language consistency reward (slight performance trade-off)
- Remaining issue: Non-English/Chinese queries may receive English responses

**Prompt Sensitivity**:
- Few-shot prompting consistently degrades performance
- Recommendation: Use zero-shot prompting with explicit output format specification

**Limited Software Engineering Scale**:
- Long evaluation times hindered RL data collection
- Result: Marginal improvements over DeepSeek-V3 on SWE tasks
- Future direction: Asynchronous evaluations during RL

### 5.2 Unsuccessful Attempts (Section 4.2)

#### Process Reward Models (PRMs)
**Challenges**:
1. Difficult to define fine-grained reasoning steps in general domains
2. Determining intermediate step correctness is hard (model-based annotation unreliable, human annotation doesn't scale)
3. Inevitable reward hacking with model-based PRMs
4. Additional computational overhead for retraining

**Conclusion**: PRMs useful for reranking but limited advantage during large-scale RL training.

#### Monte Carlo Tree Search (MCTS)
**Approach Attempted**:
- Break answers into smaller parts for systematic exploration
- Guide search with pre-trained value model
- Iteratively train actor and value models

**Fundamental Challenges**:
1. **Exponentially larger search space** than chess (token generation vs. board states)
2. **Local optima**: Extension limits cause models to get stuck
3. **Value model quality**: Fine-grained value estimation is inherently difficult for token generation
4. **Inability to self-improve**: Unlike AlphaGo's value-driven improvement loop

**Conclusion**: MCTS can boost inference performance with pre-trained value models, but iterative self-improvement remains unsolved.

---

## 6. Key Takeaways

### For Researchers

1. **RL Sufficiency**: Supervised data is not necessary for reasoning capability development—pure RL with outcome rewards suffices.

2. **Emergent Complexity**: Sophisticated behaviors (reflection, verification, meta-cognition) emerge spontaneously from simple reward signals.

3. **Distillation Effectiveness**: For models <100B parameters, distilling from stronger models outperforms direct RL training.

4. **Template Design Matters**: Minimal structural constraints allow natural capability emergence during RL.

5. **Reward Simplicity**: Rule-based rewards avoid neural model pitfalls (hacking, retraining costs) for deterministic tasks.

### For Practitioners

1. **Cost Optimization**: GRPO reduces RL costs by ~50% by eliminating the critic model.

2. **Multi-Stage Pipeline**: Combining cold-start SFT, reasoning-focused RL, rejection sampling SFT, and comprehensive RL yields production-ready models.

3. **Evaluation Robustness**: Use temperature sampling with pass@k instead of greedy decoding for stable reasoning model evaluation.

4. **Zero-Shot Prompting**: Avoid few-shot examples when using reasoning models—they degrade performance.

5. **Distillation Pathway**: For resource-constrained settings, distill from frontier models rather than training smaller models from scratch.

### For the Field

**Open Questions**:
- Can MCTS-based iterative improvement be made practical for LLMs?
- How to scale RL to software engineering tasks with expensive evaluation?
- Can process rewards be designed to avoid hacking while scaling effectively?

**Released Artifacts**:
- DeepSeek-R1-Zero (pure RL model)
- DeepSeek-R1 (production model + API)
- 6 distilled models: 1.5B, 7B, 8B, 14B, 32B, 70B (Qwen and Llama families)

---

## 7. Conclusion

DeepSeek-R1 demonstrates that **reinforcement learning alone can develop world-class reasoning capabilities** in LLMs, achieving performance competitive with OpenAI-o1-1217 on mathematical and coding benchmarks. The research challenges the prevailing assumption that large supervised datasets are necessary for reasoning, instead showing that appropriate incentive structures (via RL) enable autonomous capability development.

The "aha moment" phenomenon—where models spontaneously develop meta-cognitive reflection—represents a profound insight: **we don't need to teach models how to reason; we just need to incentivize them to find correct answers**. The sophisticated reasoning strategies emerge as instrumental behaviors for maximizing reward.

For the broader AI research community, DeepSeek's release of R1-Zero, R1, and distilled models provides unprecedented transparency into reasoning model development, enabling further research on test-time compute scaling, knowledge distillation, and emergent capabilities in language models.

**Impact**: This work validates that the path to advanced reasoning lies not in ever-larger supervised datasets, but in scalable reinforcement learning with well-designed reward structures—a finding that may fundamentally reshape how the field approaches post-training for future LLMs.

---

## References

Key citations from the paper:
- Shao et al., 2024: Group Relative Policy Optimization (GRPO)
- Lightman et al., 2023: Process-based reward models
- Silver et al., 2017: AlphaGo and AlphaZero (MCTS inspiration)
- OpenAI, 2024: o1 series models (baseline comparisons)
- Qwen, 2024: Qwen2.5 and QwQ-32B-Preview (distillation base models)

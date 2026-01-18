Reference: https://www.arxiv.org/pdf/2601.07372


# Engram: Introducing Conditional Memory as a New Axis of Sparsity for LLMs

## Executive Summary

The document presents **Engram**, a novel architecture that introduces "conditional memory" as a complementary sparsity axis to the standard "conditional computation" found in Mixture-of-Experts (MoE) models. While Transformers currently lack a native primitive for knowledge lookup—forcing them to inefficiently simulate retrieval through computation—Engram revisits classic N-gram models to provide O(1) lookups for static knowledge.

Developed by researchers at Peking University and DeepSeek-AI, this approach demonstrates that properly designed lookup mechanisms are essential complements to neural computation. By formulating the **Sparsity Allocation problem**, the authors identify a U-shaped scaling law that optimizes the trade-off between neural computation (MoE) and static memory (Engram). 

Scaling Engram to a 27B-parameter model resulted in superior performance over strict iso-parameter and iso-FLOPs MoE baselines. Crucially, the improvements extend beyond knowledge-intensive tasks to general reasoning and code/math domains. Furthermore, Engram significantly enhances long-context retrieval by delegating local dependencies to lookups, thereby freeing attention capacity for global context.

## Technical Analysis

### The Dual Nature of Language Processing
The paper posits that language modeling involves two qualitatively different sub-tasks:
1.  **Compositional Reasoning:** Requires deep, dynamic computation.
2.  **Knowledge Retrieval:** Involves local, static, and highly stereotyped patterns (e.g., named entities, formulaic patterns).

Current LLMs rely on Transformers and MoE to handle both. While MoE scales capacity effectively via conditional computation, it is still inefficient for static retrieval. For example, resolving a common multi-token entity consumes multiple early layers of attention and feed-forward networks—a process the authors describe as "expensive runtime reconstruction of a static lookup table."

### The Engram Architecture
Engram is a conditional memory module designed to structurally separate static pattern storage from dynamic computation. It operates in two main phases:

#### 1. Sparse Retrieval via Hashed N-grams
To map local contexts to static memory entries, Engram employs:
*   **Tokenizer Compression:** A surjective function maps raw token IDs to canonical identifiers (normalized textual equivalence), reducing vocabulary size by ~23% and maximizing semantic density.
*   **Multi-Head Hashing:** Since parameterizing the full space of N-grams is intractable, Engram uses a deterministic multiplicative-XOR hash function. For each N-gram order ($n$), $K$ distinct hash heads map the compressed context to indices within embedding tables ($E_{n,k}$). The final memory vector is the concatenation of all retrieved embeddings.

#### 2. Context-Aware Gating
Static embeddings lack contextual adaptability and may contain noise from hash collisions. To address this, Engram uses the current hidden state ($h_t$) as a dynamic Query, while the retrieved memory ($e_t$) serves as the source for Key and Value projections. A scalar gate ($\alpha_t$) is computed using RMSNorm:
$$ \alpha_t = \sigma\left(\frac{\text{RMSNorm}(h_t)^\top \text{RMSNorm}(k_t)}{\sqrt{d}}\right) $$
If the retrieved memory contradicts the current context, the gate suppresses the noise. A short depthwise causal convolution (kernel size 4) is then applied to expand the receptive field.

### System Efficiency and Offloading
Unlike MoE's dynamic routing, Engram employs deterministic IDs based on input tokens. This predictability enables a critical system optimization:
*   **Inference Offloading:** Embedding tables can be offloaded to host memory (CPU RAM).
*   **Prefetching:** Because indices are known before the forward pass, the host asynchronously prefetches embeddings, overlapping communication with the on-device computation of preceding Transformer blocks. Empirical results show that offloading a 100B-parameter table incurs negligible overhead (< 3%).

### Sparsity Allocation: The U-Shaped Scaling Law
The authors define the **Sparsity Allocation problem**: Given a fixed total parameter budget ($P_{tot}$) and activated parameter budget ($P_{act}$), how should "free" parameters ($P_{sparse}$) be distributed between MoE experts and Engram memory?

*   **Findings:** Experiments across different compute regimes (2e20 and 6e20 FLOPs) revealed a consistent **U-shaped relationship**.
*   **Optimal Ratio:** Pure MoE ($\rho=100\%$) is suboptimal. Reallocating roughly 20%–25% of the sparse parameter budget to Engram ($\rho \approx 75\%–80\%$) yields the best validation loss.
*   **Interpretation:**
    *   **MoE-Dominated:** Lacks dedicated memory for static patterns, forcing inefficient reconstruction via depth.
    *   **Engram-Dominated:** Loses conditional computation capacity required for dynamic, context-dependent reasoning.

### Mechanistic Analysis: Effective Depth and Context
Using **LogitLens** and **Centered Kernel Alignment (CKA)**, the paper provides insights into why Engram works:
1.  **Accelerated Prediction Convergence:** Engram models show lower KL divergence in early layers compared to MoE, indicating they reach "prediction-ready" states faster.
2.  **Increased Effective Depth:** Representational alignment analysis shows that shallow layers in Engram models semantically correspond to deeper layers in MoE baselines. Engram effectively "deepens" the network by bypassing early-stage static reconstruction.
3.  **Global Context Focus:** By handling local dependencies (e.g., entity recognition) via lookups, Engram frees the attention mechanism to focus on global context. This results in significant gains in long-context tasks like **Multi-Query NIAH** (Needle-in-a-Haystack), where Engram-27B achieved 97.0 accuracy vs. the MoE baseline's 84.2.

## Key Takeaways

*   **Complementary Sparsity:** Conditional memory (Engram) and conditional computation (MoE) are structurally complementary. Using only MoE neglects the efficiency of static lookups for local knowledge.
*   **U-Shaped Scaling Law:** There is an optimal allocation ratio between neural compute and memory capacity. For the studied models, allocating ~20-25% of sparse parameters to Engram provided the best trade-off.
*   **Beyond Knowledge Retrieval:** While intuitively beneficial for knowledge tasks (MMLU, CMMLU), Engram provided even larger relative gains in general reasoning (BBH, ARC-C) and code/math (HumanEval, MATH).
*   **Long-Context Enhancement:** Engram substantially improves long-context retrieval accuracy, likely by offloading local pattern matching to memory, allowing attention heads to specialize in global dependencies.
*   **Infrastructure-Aware Design:** Deterministic addressing allows massive embedding tables to reside in host memory with negligible latency, effectively bypassing GPU HBM constraints.

## Conclusion

The Engram architecture represents a significant shift in LLM design, re-establishing classical N-gram embeddings as a modern, scalable primitive for next-generation sparse models. By decoupling static knowledge retrieval from dynamic reasoning, Engram not only improves efficiency in knowledge-intensive domains but also enhances the model's capacity for complex reasoning and long-context understanding. The proposed U-shaped scaling law provides a guiding principle for future model scaling, suggesting that the next frontier of AI performance lies in the intelligent allocation of resources between compute and memory.

---

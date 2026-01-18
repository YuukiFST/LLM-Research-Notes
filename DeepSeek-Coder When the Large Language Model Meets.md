# DeepSeek-Coder: Technical Analysis and Documentation

## Executive Summary

DeepSeek-Coder represents a significant milestone in open-source code intelligence, introducing a family of models (1.3B, 6.7B, and 33B parameters) that achieve state-of-the-art performance among open-source code LLMs and surpass GPT-3.5-Turbo on multiple benchmarks. Trained from scratch on 2 trillion tokens spanning 87 programming languages, these models demonstrate that carefully engineered data pipelines and training strategies can bridge the gap between proprietary and open-source code models.

**Key Achievements:**
- **Performance**: DeepSeek-Coder-Base 33B achieves 50.3% average on multilingual HumanEval and 66.0% on MBPP
- **Efficiency**: The 6.7B model outperforms CodeLlama 34B despite being 5× smaller
- **Innovation**: First code LLM to implement repository-level dependency parsing during pretraining
- **Accessibility**: Released under permissive license for both research and commercial use

---

## 1. Data Engineering: Repository-Level Construction Pipeline

### 1.1 Novel Approach to Code Data Organization

DeepSeek-Coder introduces a paradigm shift from file-level to **repository-level pretraining data construction**. Unlike previous models (CodeLlama, StarCoder) that treat files independently, this approach preserves cross-file dependencies critical for real-world coding scenarios.

**Data Composition:**
- 87% source code (798 GB, 603M files)
- 10% English code-related corpus (GitHub Markdown, StackExchange)
- 3% Chinese natural language corpus

### 1.2 Five-Stage Data Pipeline

#### Stage 1: GitHub Crawling and Rule-Based Filtering

**Scope**: Public repositories created before February 2023 across 87 programming languages

**Filtering Criteria** (adapted from StarCoder):
- Average line length < 100 characters
- Maximum line length < 1000 characters
- Minimum 25% alphabetic characters
- HTML files: visible text ≥ 20% of code and ≥ 100 characters
- JSON/YAML: 50-5000 character range

**Impact**: Reduced dataset to 32.8% of original size while maintaining quality

#### Stage 2: Dependency Parsing via Topological Sort

**Core Innovation**: Files within a repository are arranged in dependency order, ensuring that imported/included files appear before files that depend on them.

**Algorithm** (Algorithm 1 in paper):
1. Build dependency graph using regex extraction of import statements:
   - Python: `import`
   - C#: `using`
   - C: `include`
2. Apply modified topological sort that handles cycles by selecting nodes with minimal (not zero) in-degrees
3. Concatenate files in dependency order with path comments as headers

**Training Sample Structure**:
```
# /path/to/base_module.py
[base module code]

# /path/to/dependent_module.py
[code that imports base_module]
```

**Benefit**: Models learn realistic code patterns where dependencies precede usage, improving cross-file code completion by 2-3% (Table 7).

#### Stage 3: Repository-Level Deduplication

**Key Distinction**: Deduplication performed at repository granularity rather than file level to preserve structural integrity.

**Rationale**: File-level deduplication may remove files from a repository, breaking dependency chains and disrupting the repository's logical structure.

**Implementation**: Concatenated repository content treated as single sample for near-deduplication algorithm.

#### Stage 4: Quality Screening

**Multi-pronged approach**:
- Compiler-based validation for syntax errors
- Quality model scoring for readability and modularity
- Heuristic rules for code style

**Final Statistics**: 87 languages, 798 GB, 603M files (Table 1 shows Java at 18.63%, Python at 15.12%, C++ at 11.39% as top languages)

#### Stage 5: Decontamination

**N-gram filtering** to prevent test set leakage:
- 10-gram exact match for strings ≥10 tokens
- 3-9 gram exact match for shorter strings
- Applied to HumanEval, MBPP, GSM8K, MATH benchmarks

---

## 2. Training Architecture and Methodology

### 2.1 Dual-Objective Training Strategy

#### Objective 1: Next Token Prediction
Standard autoregressive training on concatenated file sequences.

#### Objective 2: Fill-in-the-Middle (FIM)

**Motivation**: Code completion requires bidirectional context (prefix + suffix) to generate middle content accurately.

**Implementation**:
- Three sentinel tokens: `<|fim_start|>`, `<|fim_hole|>`, `<|fim_end|>`
- PSM (Prefix-Suffix-Middle) mode:
  ```
  <|fim_start|> f_pre <|fim_hole|> f_suf <|fim_end|> f_middle <|eos_token|>
  ```

**Critical Ablation Study (Figure 3)**:

Testing four configurations on DeepSeek-Coder 1.3B:
- **0% FIM**: Baseline next-token prediction only
- **50% PSM**: Optimal balance (selected configuration)
- **100% FIM**: Peak FIM performance but degraded completion
- **50% MSP**: Masked Span Prediction (T5-style)

**Finding**: 50% PSM rate achieves best trade-off between FIM capability and code completion performance. 100% FIM creates specialization that hurts general code generation.

### 2.2 Model Architecture

**Base Framework**: DeepSeek LLM decoder-only Transformer

| Hyperparameter | 1.3B | 6.7B | 33B |
|---|---|---|---|
| Hidden Size | 2048 | 4096 | 7168 |
| Layers | 24 | 32 | 62 |
| Attention Heads | 16 | 32 | 56 (GQA-8) |
| Attention Type | Multi-head | Multi-head | Grouped-Query |
| Batch Size | 1024 | 2304 | 3840 |
| Max Learning Rate | 5.3e-4 | 4.2e-4 | 3.5e-4 |

**Key Components**:
- **RoPE** (Rotary Position Embedding) for positional encoding
- **SwiGLU** activation function
- **FlashAttention v2** for computational efficiency
- **GQA** (Grouped-Query Attention) in 33B model for improved inference speed

**Tokenizer**: BPE with 32,000 vocabulary trained on corpus subset

### 2.3 Optimization Strategy

**Optimizer**: AdamW (β₁=0.9, β₂=0.95)

**Three-Stage Learning Rate Schedule**:
1. 2000 warmup steps
2. Each stage scaled by √(1/10) from previous stage
3. Final LR = 10% of initial rate

**Infrastructure**:
- HAI-LLM framework with tensor, ZeRO data, and PipeDream pipeline parallelism
- NVIDIA A100 and H800 GPU clusters with NVLink/NVSwitch and InfiniBand interconnects

### 2.4 Context Window Extension

**Baseline**: 16K context window

**Extended Configuration** (for repository-level code):
- RoPE scaling factor: 1 → 4
- Base frequency: 10,000 → 100,000
- Additional 1000 training steps with 16K sequences
- **Theoretical capacity**: 64K tokens
- **Reliable output range**: 16K tokens

### 2.5 Instruction Tuning

**DeepSeek-Coder-Instruct** created via supervised fine-tuning:
- Alpaca instruction format with `<|EOT|>` delimiter
- Cosine schedule, 100 warmup steps, LR=1e-5
- Batch size: 4M tokens, Total: 2B tokens

**Multi-turn Capability**: Demonstrated in Figure 4 (snake game example) with iterative feature additions and code explanations.

---

## 3. Experimental Results and Benchmarks

### 3.1 Code Generation Performance

#### HumanEval Multilingual (8 languages)

**DeepSeek-Coder-Base 33B Results**:
| Language | Score | vs CodeLlama-34B |
|---|---|---|
| Python | 56.1% | +7.9% |
| C++ | 58.4% | +13.7% |
| Java | 51.9% | +7.0% |
| JavaScript | 55.3% | +13.1% |
| **Average** | **50.3%** | **+9.3%** |

**Key Finding**: DeepSeek-Coder 6.7B (44.7% avg) outperforms CodeLlama 34B (41.0% avg) despite being 5× smaller—evidence of superior data quality.

#### MBPP (Python)
- **DeepSeek-Coder-Base 33B**: 66.0% (vs CodeLlama-34B: 55.2%)
- **DeepSeek-Coder-Instruct 33B**: 70.0% (vs GPT-3.5: 70.8%, approaching parity)

#### DS-1000 (Real-World Data Science)

Testing library-specific code generation across 7 libraries:

| Model | Matplotlib | NumPy | Pandas | PyTorch | Avg |
|---|---|---|---|---|---|
| CodeLlama-34B | 50.3% | 42.7% | 23.0% | 25.0% | 34.3% |
| **DeepSeek-Coder-33B** | **56.1%** | **49.6%** | **25.8%** | **36.8%** | **40.2%** |

**Improvement**: +5.9% absolute over CodeLlama-34B, demonstrating practical applicability.

#### LeetCode Contest (Competition Problems)

180 problems from July 2023–January 2024 (post-training cutoff):

| Model | Easy | Medium | Hard | Overall |
|---|---|---|---|---|
| GPT-3.5-Turbo | 46.7% | 15.4% | 15.9% | 23.3% |
| **DeepSeek-Coder-Instruct 33B** | **57.8%** | **22.0%** | **9.1%** | **27.8%** |
| GPT-4-Turbo | 73.3% | 31.9% | 25.0% | 40.6% |

**Chain-of-Thought Impact**: Adding "write a step-by-step outline first" improves 33B model from 27.8% → 28.9%, with gains concentrated in Medium difficulty (+3.3%).

### 3.2 Fill-in-the-Middle Code Completion

**Single-Line Infilling Benchmark** (3 languages):

| Model | Size | Python | Java | JavaScript | Mean |
|---|---|---|---|---|---|
| StarCoder | 16B | 62.0% | 73.0% | 74.0% | 69.7% |
| CodeLlama-13B | 13B | 68.3% | 77.6% | 80.7% | 75.5% |
| **DeepSeek-Coder 6.7B** | **6.7B** | **66.6%** | **88.1%** | **79.7%** | **80.7%** |

**Remarkable**: 1.3B model achieves 70.4% mean, competitive with StarCoder-16B (69.7%).

### 3.3 Cross-File Code Completion

**CrossCodeEval Benchmark** (March–June 2023 repositories):

**Exact Match (EM) scores with BM25 retrieval**:

| Model | Python | Java | TypeScript | C# | Avg |
|---|---|---|---|---|---|
| CodeLlama-7B + Retrieval | 13.02% | 16.41% | 12.34% | 13.19% | 13.74% |
| **DeepSeek-Coder 6.7B + Retrieval** | **16.14%** | **17.72%** | **14.03%** | **16.23%** | **16.03%** |

**Ablation**: Removing repository-level pretraining reduces performance by 1-2% across Java/TypeScript/C#, validating the dependency parsing approach.

### 3.4 Program-Based Math Reasoning

**PAL (Program-Aided Language Models) on 7 benchmarks**:

| Model | GSM8K | MATH | TabMWP | MAWPS | Avg |
|---|---|---|---|---|---|
| CodeLlama-34B | 58.2% | 21.2% | 69.8% | 91.8% | 62.0% |
| **DeepSeek-Coder-33B** | **60.7%** | **29.1%** | **75.3%** | **93.3%** | **65.8%** |

**MATH benchmark improvement**: +7.9% absolute, indicating stronger mathematical reasoning via code generation.

---

## 4. DeepSeek-Coder v1.5: Enhanced Natural Language Understanding

### 4.1 Continued Pretraining Strategy

**Motivation**: Code models must understand natural language instructions to be practical.

**Approach**: Additional 2T token training starting from DeepSeek-LLM-7B Base checkpoint (general LLM).

**Data Mix**:
- 70% source code
- 10% Markdown/StackExchange
- 7% code-related natural language
- 7% math-related natural language
- 6% bilingual (Chinese-English) text

**Change**: 4K context (vs 16K in base), next-token prediction only (no FIM).

### 4.2 Performance Comparison

**DeepSeek-Coder v1.5 7B vs DeepSeek-Coder 6.7B**:

| Category | Task | v1.5 7B | Base 6.7B | Change |
|---|---|---|---|---|
| **Programming** | HumanEval | 43.2% | 44.7% | -1.5% |
| | MBPP | 60.4% | 60.6% | -0.2% |
| **Math** | GSM8K | 62.4% | 43.2% | **+19.2%** |
| | MATH | 24.7% | 19.2% | **+5.5%** |
| **NLP** | MMLU | 49.1% | 36.6% | **+12.5%** |
| | BBH | 55.2% | 44.3% | **+10.9%** |
| | HellaSwag | 69.9% | 53.8% | **+16.1%** |

**Key Insight**: Slight coding regression (-1-2%) yields massive gains in math reasoning (+19%) and general language understanding (+10-16%), demonstrating the value of building code models atop strong general LLMs.

---

## 5. Key Takeaways and Lessons Learned

### 5.1 Data Quality Over Model Size

**Evidence**: DeepSeek-Coder 6.7B outperforms CodeLlama 34B across most benchmarks.

**Implication**: Repository-level organization, rigorous deduplication, and quality screening create more efficient learning than simply scaling data volume.

### 5.2 Repository-Level Training Matters

**Cross-file completion gains**: 2-3% improvement with dependency parsing (Table 7 ablation).

**Real-world relevance**: Professional coding involves multi-file projects; file-level training creates artificial boundaries.

### 5.3 FIM Training Requires Careful Tuning

**Critical finding**: 100% FIM rate maximizes infilling but degrades general completion.

**Optimal configuration**: 50% PSM balances both capabilities without specialization penalties.

**Design principle**: Multi-objective training requires empirical optimization of objective mixing ratios.

### 5.4 Instruction Tuning Unlocks Practical Utility

**Base → Instruct gains**:
- HumanEval Python: 56.1% → 79.3% (+23.2%)
- LeetCode: Not applicable → 27.8% (surpasses GPT-3.5)

**Multi-turn capability**: Enables iterative development workflows (Figure 4 snake game example).

### 5.5 General LLM Foundation Enables Broader Capabilities

**v1.5 results**: Building on DeepSeek-LLM checkpoint creates models that:
- Maintain code generation competence
- Add strong math reasoning (+19% GSM8K)
- Understand natural language instructions better (+12% MMLU)

**Future direction**: "The most effective code-focused LLMs are built upon robust general LLMs" (Section 6).

---

## 6. Limitations and Future Work

### 6.1 Acknowledged Limitations

1. **Context window reliability**: 16K empirically reliable despite 64K theoretical capacity
2. **Data contamination risk**: LeetCode benchmark scores higher in July-August contests
3. **Evaluation coverage**: Limited benchmarks for some of 87 supported languages

### 6.2 Architectural Constraints

- **33B model**: Requires high-end GPUs (A100/H800) for practical deployment
- **FIM specialization**: Trade-off between completion and infilling remains
- **Multilingual balance**: Top 3 languages (Java, Python, C++) dominate 45% of training data

### 6.3 Research Directions

1. **Long-context enhancement**: Improve reliability beyond 16K tokens for large codebases
2. **Benchmark development**: Create contamination-resistant evaluation sets
3. **Scaling experiments**: Train larger models (67B+) on general LLM foundations
4. **Domain adaptation**: Specialized versions for security, embedded systems, etc.

---

## 7. Conclusion and Impact

### 7.1 Research Contributions

DeepSeek-Coder advances the state of open-source code intelligence through:

1. **Methodological innovation**: Repository-level dependency parsing during pretraining
2. **Empirical insights**: FIM training configuration ablations (Figure 3)
3. **Performance milestones**: First open-source model to surpass GPT-3.5-Turbo on multiple code benchmarks
4. **Accessibility**: Permissive licensing enables commercial deployment and research

### 7.2 Practical Impact

**For researchers**:
- Comprehensive benchmark suite (HumanEval, MBPP, DS-1000, LeetCode, CrossCodeEval)
- Reproducible training methodology with published hyperparameters
- Open-source weights enabling analysis and extensions

**For practitioners**:
- Production-ready models from 1.3B (edge devices) to 33B (cloud services)
- Instruction-tuned variants for chat/agent workflows
- Strong performance on real-world tasks (DS-1000 library usage, LeetCode competitions)

### 7.3 Broader Implications

**Democratization of AI**: By matching GPT-3.5 performance in open-source form, DeepSeek-Coder reduces dependency on proprietary APIs and enables:
- On-premise deployment for sensitive codebases
- Fine-tuning for domain-specific applications
- Research without API cost barriers

**Validation of data-centric approach**: Demonstrates that careful data engineering (repository structure, deduplication, quality screening) can outperform naive scaling—a lesson applicable beyond code models.

**Foundation for future work**: v1.5 experiments validate the "general LLM → specialized model" paradigm, suggesting a two-stage development process for domain-specific models.

---

## 8. Technical Specifications Summary

### 8.1 Model Variants

| Variant | Parameters | Context | FIM Rate | Use Case |
|---|---|---|---|---|
| DeepSeek-Coder-Base 1.3B | 1.3B | 16K | 50% | Edge devices, fast inference |
| DeepSeek-Coder-Base 6.7B | 6.7B | 16K | 50% | Code completion tools |
| DeepSeek-Coder-Base 33B | 33B | 16K | 50% | Advanced code generation |
| DeepSeek-Coder-Instruct (all sizes) | Same | 16K | N/A | Chat, instruction-following |
| DeepSeek-Coder-v1.5 7B | 6.9B | 4K | 0% | General-purpose coding assistant |

### 8.2 Training Corpus Statistics

- **Total size**: 798 GB, 603M files
- **Languages**: 87 programming languages
- **Tokens**: 2 trillion (pretraining)
- **Composition**: 87% code, 10% English code-related, 3% Chinese
- **Temporal coverage**: GitHub repositories created before February 2023

### 8.3 Benchmark Performance Summary

**Best-in-class results** (DeepSeek-Coder-Base 33B):
- Multilingual HumanEval: 50.3% average (9% over CodeLlama-34B)
- MBPP: 66.0% (11% over CodeLlama-34B)
- DS-1000: 40.2% (6% over CodeLlama-34B)
- CrossCodeEval: 16.03% EM average (2% over CodeLlama-7B)

**Instruction-tuned achievements** (DeepSeek-Coder-Instruct 33B):
- Surpasses GPT-3.5-Turbo on HumanEval (69.2% vs 64.9%)
- LeetCode contests: 27.8% overall (4.5% over GPT-3.5)
- Narrows gap to GPT-4 (40.6% → 27.8% = 12.8% gap)

---

## Appendix: Reproducibility Information

### A.1 Data Processing Pipeline

```
GitHub Crawl → Rule Filtering (67% reduction) → Dependency Parsing 
→ Repo-level Deduplication → Quality Screening → Decontamination
```

### A.2 Training Configuration

**Hardware**: NVIDIA A100/H800 clusters with InfiniBand
**Framework**: HAI-LLM (parallelism: tensor + ZeRO + PipeDream)
**Total training time**: ~2000 steps warmup + 3-stage schedule
**Validation**: 8000 code files sampled from training corpus

### A.3 Evaluation Methodology

- **Greedy decoding**: All benchmarks use deterministic generation
- **Consistent environment**: Baselines re-implemented in same framework
- **Pass@1 metric**: Single attempt success rate (no sampling)
- **Test contamination**: N-gram filtering applied to training data

### A.4 Public Resources

- **Code**: https://github.com/deepseek-ai/DeepSeek-Coder
- **Models**: Available via HuggingFace model hub
- **License**: Permissive (research + commercial use)
- **LeetCode benchmark**: https://github.com/deepseek-ai/DeepSeek-Coder/tree/main/Evaluation/LeetCode

---
You're right! I should include a proper references section. Here's the complete addition:

---

## References & Resources

### Primary Resources
- **Paper**: DeepSeek-Coder: When the Large Language Model Meets Programming - The Rise of Code Intelligence (arXiv:2401.14196v2)
- **Code Repository**: https://github.com/deepseek-ai/DeepSeek-Coder
- **Authors**: Guo et al. (DeepSeek-AI & Peking University)
- **License**: Permissive (research and commercial use)

### Model Repositories
- **HuggingFace Models**: DeepSeek-Coder-Base and DeepSeek-Coder-Instruct (1.3B, 6.7B, 33B variants)
- **LeetCode Benchmark**: https://github.com/deepseek-ai/DeepSeek-Coder/tree/main/Evaluation/LeetCode

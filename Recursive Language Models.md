# Recursive Language Models

## Executive Summary

Recursive Language Models (RLMs) introduce a general-purpose inference paradigm enabling LLMs to process inputs up to two orders of magnitude beyond their context window limits by treating prompts as external environment variables. The approach initializes a Python REPL environment where the prompt is stored as a string variable, allowing the model to programmatically inspect, decompose, and recursively invoke sub-LLM calls over prompt snippets. Experimental validation using GPT-5 and Qwen3-Coder-480B-A35B across four diverse long-context benchmarks demonstrates that RLMs maintain strong performance on inputs exceeding 10M tokens, outperforming base models and common scaffolds by margins of 12-91 percentage points on information-dense tasks while maintaining comparable or lower inference costs. On the quadratic-complexity OOLONG-Pairs benchmark, RLMs achieve F1 scores of 58.00% (GPT-5) and 23.11% (Qwen3-Coder) versus <0.1% for base models, representing an emergent capability for extremely information-dense reasoning.

---

## 1. Technical Architecture

### 1.1 Motivation and Problem Statement

Modern language models exhibit context rot—performance degradation as input length increases—even within stated context windows. Frontier models like GPT-5 with 272K token windows demonstrate rapid quality decline on tasks requiring dense access to extended contexts. Existing inference-time approaches such as context compaction presume early prompt details can be safely discarded, failing on tasks requiring comprehensive information access. The problem formulation addresses whether general-purpose LLMs can scale context size by orders of magnitude through inference-time computation, motivated by increasing adoption for long-horizon tasks processing hundreds of millions of tokens.

### 1.2 Architecture Components

RLMs expose an identical external interface to standard LLMs (string prompt input, string response output) while fundamentally altering internal processing. Given prompt P, the system:

1. **Environment Initialization**: Instantiates a Read-Eval-Print Loop (REPL) programming environment with P assigned as a variable value
2. **Context Exposure**: Provides the LLM metadata about the environment (e.g., string length of P) without direct token-level ingestion
3. **Programmatic Interaction**: Enables code generation for inspecting and decomposing P with execution feedback observation
4. **Recursive Invocation**: Permits construction of sub-tasks invoking the model recursively over programmatic prompt snippets

The REPL environment provides:
- `context` variable containing the full prompt as a manipulable string
- `llm_query(prompt, context_snippet)` function for recursive sub-LLM invocation
- `print()` statements for iterative reasoning and observation
- Standard Python execution capabilities for symbolic manipulation

Key architectural distinction: prompts exist as symbolic environment objects rather than neural network inputs, enabling unbounded scaling beyond transformer context windows.

### 1.3 Training or Pre-Training Protocol

Not applicable per source document. RLMs operate as an inference-time strategy applied to pre-trained frontier models without additional training.

### 1.4 Performance Impact

**Scaling Characteristics**:
- Successfully handles inputs 100× beyond model context windows (272K → 10M+ tokens)
- Maintains performance where base models degrade catastrophically
- Cost scales proportionally to task complexity while remaining within same order of magnitude as base model calls

**Efficiency Metrics**:
- BrowseComp-Plus (6-11M tokens): RLM average cost $0.99 vs theoretical base model cost $1.50-$2.75
- Median RLM inference cost cheaper than median base model on evaluated tasks
- High variance in trajectory length creates tail-end cost increases (95th percentile significantly higher)
- Sequential LM call implementation creates runtime bottlenecks addressable through asynchronous execution

**Quality-Cost Tradeoffs**:
- Information-dense tasks show 10-59% performance gains over ablations without sub-calling
- Tradeoff point exists where small inputs favor base models over RLM overhead
- Sub-call granularity significantly impacts cost: model-specific tendencies (Qwen3-Coder uses 1000× more sub-calls than GPT-5 on identical tasks)

---

## 2. Post-Training or Optimization Methods

No post-training optimization methods described. The system operates through prompt engineering and environmental scaffolding applied at inference time. Key prompt components include:

**System Prompt Structure** (fixed across experiments):
- Environment description with context metadata (type, length, chunk boundaries)
- Function specifications for `llm_query` and execution primitives
- Strategic guidance on chunking, filtering, and recursive decomposition
- Output format requirements using `FINAL()` and `FINAL_VAR()` tags

**Model-Specific Adaptations**:
- Qwen3-Coder prompt includes rate-limiting guidance: "Be very careful about using 'llm_query' as it incurs high runtime costs. Always batch as much information as reasonably possible into each call (aim for ~200k characters per call)."
- GPT-5 prompt encourages liberal sub-LM usage without explicit rate constraints
- No task-specific prompt tuning performed

---

## 3. Agentic or System-Level Design

RLMs demonstrate emergent agentic behaviors without explicit workflow engineering:

**Context Management Patterns**:
1. **Filtering via model priors**: Uses regex and keyword searches informed by world knowledge (e.g., searching for "festival" and domain-specific terms like "La Union") to narrow context before LLM inspection
2. **Chunking strategies**: Employs uniform chunking or keyword-based segmentation, typically processing 10 documents per sub-LLM call or splitting by natural boundaries (markdown headers, newlines)
3. **Answer verification**: Implements multi-step verification through redundant sub-LLM calls on small context windows, though this sometimes increases cost without accuracy benefit

**Recursive Decomposition**:
- Sub-calls limited to depth=1 in evaluated configuration (sub-LMs are base models, not recursive)
- Task decomposition deferred to model reasoning rather than human-engineered workflows
- Information aggregation through programmatic variable manipulation and progressive refinement

**Long Output Generation**:
- Circumvents output token limits by storing sub-LLM responses in REPL variables
- Constructs final outputs through programmatic stitching of recursive call results
- Enables unbounded output length through environmental buffering

**Model-Specific Behaviors**:
- GPT-5: Conservative sub-calling (tens of calls per task), strategic chunking
- Qwen3-Coder: Aggressive sub-calling (hundreds to thousands per task), line-by-line semantic processing
- Both models exhibit non-optimal decision-making including redundant verification loops and answer regeneration

---

## 4. Benchmark Performance and Ablations

### Primary Results (Table 1 Reproduction)

| Model | Method | CodeQA | BrowseComp+ (1K) | OOLONG | OOLONG-Pairs |
|-------|--------|--------|------------------|---------|--------------|
| **Qwen3-Coder-480B** | Base Model | 20.00* | 0.00* | 36.00 | 0.06 |
| | CodeAct + BM25 | 24.00* | 12.66 | 38.00 | 0.28 |
| | Summary Agent | 50.00 | 38.00 | 44.06 | 0.31 |
| | RLM | **56.00** | **44.66** | **48.00** | **23.11** |
| | RLM (no sub-calls) | 66.00 | 46.00 | 43.50 | 17.34 |
| **GPT-5** | Base Model | 24.00* | 0.00* | 44.00 | 0.04 |
| | CodeAct + BM25 | 22.00* | 51.00 | 38.00 | 24.67 |
| | Summary Agent | 58.00 | 70.47 | 46.00 | 0.01 |
| | RLM | **62.00** | **91.33** | **56.50** | **58.00** |
| | RLM (no sub-calls) | 58.00 | 88.00 | 36.00 | 43.93 |

*\* Indicates context window exceeded*

**Task Characteristics**:
- S-NIAH: Constant complexity (single needle regardless of length)
- CodeQA: 23K-4.2M tokens per task
- BrowseComp+: 6M-11M tokens per task
- OOLONG: 131K tokens, linear complexity
- OOLONG-Pairs: 32K tokens, quadratic complexity

### Scaling Analysis (Figure 1)

Performance across context lengths 2^13 to 2^18 tokens:

**S-NIAH (Constant Complexity)**:
- GPT-5 base: Maintains 80-100% until 2^14, degrades to 20% at 2^18
- RLM(GPT-5): Maintains 90-100% across all scales

**OOLONG (Linear Complexity)**:
- GPT-5 base: Degrades from 80% (2^13) to 20% (2^16)
- RLM(GPT-5): Degrades from 95% (2^13) to 60% (2^18), 40pp advantage at 2^16

**OOLONG-Pairs (Quadratic Complexity)**:
- GPT-5 base: Catastrophic failure <5% across all scales
- RLM(GPT-5): Maintains 60-80% performance across scales

### Ablation Studies

**REPL Environment Contribution**:
- RLM (no sub-calls) outperforms base model on all tasks fitting in context
- CodeQA with Qwen3-Coder: no-sub-calls ablation achieves 66% vs 56% for full RLM (programmatic solving sufficient)
- BrowseComp+: Minimal degradation (46% vs 44.66% for Qwen3-Coder)

**Sub-Calling Necessity**:
- Information-dense tasks show 10-59% gains with sub-calling enabled
- OOLONG: +4.5pp (Qwen3), +20.5pp (GPT-5)
- OOLONG-Pairs: +5.77pp (Qwen3), +14.07pp (GPT-5)
- Critical for semantic transformations requiring model inference versus keyword heuristics

### Cost Analysis

**Median Costs**:
- GPT-5 RLM median cost lower than base model across most tasks
- Qwen3-Coder shows higher variance due to aggressive sub-calling

**Cost Scaling** (BrowseComp-Plus 1K documents):
- Summary Agent (Qwen3): $8.98 ± $2.12
- RLM (Qwen3): $0.84 ± $0.63
- Summary Agent (GPT-5): $0.57 ± $0.10
- RLM (GPT-5): $0.99 ± $1.22

**95th Percentile**:
- Long-tail trajectories show 5-10× median cost due to extensive iteration

---

## 5. Key Technical Takeaways

- RLMs achieve 100× context window extension through environmental offloading without architectural modification or retraining
- Performance degradation exhibits slower decay as task complexity increases compared to base models (40-50pp advantages on quadratic-complexity tasks)
- REPL environment alone enables beyond-window scaling; recursive sub-calling provides 10-59% additional gains on information-dense tasks
- Inference cost remains within same order of magnitude as base model despite 100× input scaling, with median costs frequently lower
- Emergent behaviors include model-prior-based filtering, strategic chunking, and programmatic answer construction without explicit training
- Model-specific tendencies create orders-of-magnitude differences in sub-call granularity (GPT-5: tens, Qwen3-Coder: thousands per task)
- Trajectory optimization opportunity exists: observed redundant verification loops and non-optimal decomposition strategies
- Tradeoff point for small inputs where base model outperforms RLM by 5-10pp due to scaffolding overhead

---

## 6. Conclusion

Recursive Language Models establish a general-purpose inference paradigm enabling order-of-magnitude context scaling through symbolic prompt manipulation in external environments. By treating prompts as REPL variables rather than transformer inputs, the approach circumvents architectural context limits while maintaining inference costs comparable to base model calls. Empirical validation across four benchmarks spanning 23K to 11M tokens demonstrates 12-91pp performance advantages over existing methods, with particularly strong results on information-dense tasks where base models fail catastrophically. The framework exhibits emergent agentic capabilities including strategic filtering, recursive decomposition, and programmatic aggregation without task-specific engineering or additional training. Performance characteristics suggest RLMs provide a viable scaling axis complementary to architectural advances, with future work directions including asynchronous sub-call optimization, deeper recursion depths, and explicit training for RLM operation patterns.

---

## References

Zhang, A. L., Kraska, T., & Khattab, O. (2025). Recursive Language Models. arXiv preprint arXiv:2512.24601v1.

**Related Works Cited in Analysis**:

Anthropic. (2025). Claude code: Subagents — modular ai workflows with isolated agent contexts. https://docs.anthropic.com/en/docs/claude-code/sub-agents

Bai, Y., Tu, S., Zhang, J., et al. (2025). Longbench v2: Towards deeper understanding and reasoning on realistic long-context multitasks. arXiv:2412.15204.

Bertsch, A., Pratapa, A., Mitamura, T., Neubig, G., & Gormley, M. R. (2025). Oolong: Evaluating long context reasoning and aggregation capabilities. arXiv:2511.02817.

Chen, H., Pasunuru, R., Weston, J., & Celikyilmaz, A. (2023). Walking down the memory maze: Beyond context limit through interactive reading. arXiv:2310.05029.

Chen, Z., Ma, X., Zhuang, S., et al. (2025). Browsecomp-plus: A more fair and transparent evaluation benchmark of deep-research agent. arXiv:2508.06600.

DeepSeek-AI et al. (2025). Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv:2501.12948.

Goldman, O., Jacovi, A., Slobodkin, A., et al. (2025). Is it really long context if all you need is retrieval? Towards genuinely difficult long context nlp. arXiv:2407.00402.

Grand, G., Tenenbaum, J. B., Mansinghka, V. K., Lew, A. K., & Andreas, J. (2025). Self-steering language models. arXiv:2504.07081.

Hong, K., Troynikov, A., & Huber, J. (2025). Context rot: How context degradation affects llm performance. https://research.trychroma.com/context-rot

Hsieh, C.-P., Sun, S., Kriman, S., et al. (2024). Ruler: What's the real context size of your long-context language models? arXiv:2404.06654.

Khattab, O., Potts, C., & Zaharia, M. (2021). Baleen: Robust multi-hop reasoning at scale via condensed retrieval. Advances in Neural Information Processing Systems, 34, 27670-27682.

OpenAI. (2025). Deep research. https://openai.com/index/introducing-deep-research/

OpenAI et al. (2024). Openai o1 system card. arXiv:2412.16720.

Schroeder, P., Morgan, N., Luo, H., & Glass, J. (2025). Thread: Thinking deeper with recursive spawning. arXiv:2405.17402.

Smith, C. (2025). Openhands context condensensation for more efficient ai agents. https://openhands.dev/blog/openhands-context-condensensation-for-more-efficient-ai-agents

Sun, W., Lu, M., Ling, Z., et al. (2025). Scaling long-horizon llm agent via context-folding. arXiv:2510.11967.

Wang, X., Chen, Y., Yuan, L., et al. (2024). Executable code actions elicit better llm agents. arXiv:2402.01030.

Wu, X., Li, K., Zhao, Y., et al. (2025). Resum: Unlocking long-horizon search intelligence via context summarization. arXiv:2509.13313.

Yang, A., Li, A., Yang, B., et al. (2025). Qwen3 technical report. arXiv:2505.09388.

Yao, S., Zhao, J., Yu, D., et al. (2023). React: Synergizing reasoning and acting in language models. arXiv:2210.03629.

Ye, R., Zhang, Z., Li, K., et al. (2025). Agentfold: Long-horizon web agents with proactive context management. arXiv:2510.24699.

Zelikman, E., Wu, Y., Mu, J., & Goodman, N. D. (2022). Star: Bootstrapping reasoning with reasoning. arXiv:2203.14465.

Zhu, A., Dugan, L., & Callison-Burch, C. (2024). Redel: A toolkit for llm-powered recursive multi-agent systems. arXiv:2408.02248.

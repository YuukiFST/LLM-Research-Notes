I'll analyze this technical document about xDeepServe and produce a formal technical report following the specified format.

# xDeepServe: Model-as-a-Service on Huawei CloudMatrix384

## Executive Summary

xDeepServe is Huawei Cloud's LLM serving system designed for SuperPod-scale infrastructure, specifically targeting large Mixture-of-Experts (MoE) models on CloudMatrix384's 384-NPU architecture. The system introduces Transformerless, a disaggregated architecture that decomposes transformer models into independently executable modules (attention, feedforward, MoE) across NPUs connected via high-speed fabric. Core contributions include: (1) XCCL, a communication library leveraging global shared memory for microsecond-level point-to-point and all-to-all primitives; (2) FlowServe, a scalable serving engine with Data Parallel group abstraction eliminating single points of failure; (3) disaggregated prefill-decode and disaggregated MoE-attention execution modes. The system achieves 2400 tokens/second per NPU chip while maintaining 50ms time-per-output-token (TPOT) on DeepSeek models at production scale. Deployment spans 768 NPU dies with 288 handling MoE (EP288) and 480 handling attention, supporting global batch sizes exceeding 46,000 tokens.

---

## 1. Technical Architecture

### 1.1 Motivation and Problem Statement

Large-scale MoE models (DeepSeek, Kimi K2, Qwen) require fine-grained expert routing, synchronization, and load balancing across hundreds of NPU devices. CloudMatrix384 provides global shared memory and uniform low-latency access across 384 NPUs, but traditional LLM serving systems cannot exploit these properties. The convergence of scaled-out MoE models and scaled-up SuperPod hardware creates three challenges: (1) efficient expert load balancing under EP288 configurations; (2) elimination of single points of failure across hundreds of NPUs; (3) maintaining low tail latency under global synchronization barriers introduced by MoE dispatch and combine operations.

### 1.2 Architecture Components

**CloudMatrix384 SuperPod Architecture**

CloudMatrix384 comprises 48 servers with 384 Ascend NPU chips (768 dies total). Each NPU chip contains two dies interconnected via high-bandwidth Network-on-Chip (NoC). Three network fabrics are integrated: (1) VPC network for external connectivity; (2) RoCE network for scaled-out interconnection across SuperPods; (3) UB network providing all-to-all connectivity within the SuperPod at several times RoCE bandwidth. The UB network enables global shared memory addressing across all CPU DRAM and NPU on-chip memory, supporting both DMA and memory semantics. Intra-server NUMA locality is eliminated—CPUs have uniform access latency to all eight NPUs per server.

**Transformerless Disaggregated Architecture**

Transformerless decomposes transformer models into three modular components executed independently: attention blocks, feedforward networks (FFN), and MoE layers. Two disaggregation forms are implemented:

1. Disaggregated Prefill-Decode: Separates compute-bound prefill from memory-bound decode on different NPU sets. Prefill runs on both Ascend 910 and CloudMatrix384 NPUs; decode runs exclusively on CloudMatrix384 to leverage high-bandwidth interconnects.

2. Disaggregated MoE-Attention: Separates MoE experts from attention computation on distinct NPUs. In production deployment, 288 NPU dies execute EP288 (256 routed experts plus 32 shared experts), while 480 dies handle Multi-head Latent Attention (MLA) computation.

**XCCL Communication Library**

XCCL provides memory-semantic primitives over CloudMatrix384's distributed shared memory:

- Point-to-point send/recv: Each NPU's on-chip memory is partitioned into app data area, metadata area (32-byte fields per computing core pair), and managed data area (ring buffers per NPU pair). Send/recv protocol uses distributed memory semantics with mtu-in/mtu-out transfers and ping-pong buffers for concurrent operation.

- All-to-all dispatch/combine: For expert parallelism, metadata area contains fields per rank with event ID, buffer offset pointer, and token count. Pull-based protocol broadcasts token counts, then performs data transfer guided by received offsets.

- Attention-to-Expert (A2E) / Expert-to-Attention (E2A): Trampoline forward mechanism addresses asymmetric NPU allocation (288 MoE NPUs vs 160 attention NPUs). Subset of expert NPUs matching attention NPU count receives data first, then forwards to remaining expert NPUs in two-stage routing.

**FlowServe Serving Engine**

FlowServe implements Data Parallel (DP) group abstraction where each group encapsulates complete serving pipeline: tokenization, API parsing, SPMD executors, Relational Tensor Cache (RTC), and DistFlow networking stack. Key components are replicated within each DP to eliminate cross-DP dependencies. TE-shell orchestrator handles only request dispatch, expert load balancing triggers, and health checks. Master process spawns child process for output handling (detokenization, stream parsing) with direct relay to frontend.

### 1.3 Training or Pre-Training Protocol

Not applicable per source document. The system focuses on inference serving rather than training.

### 1.4 Performance Impact

**Communication Performance**

Point-to-point primitives achieve sub-20μs latency for payloads under 1MB using 2 computing cores. With 48 computing cores, 9MB transfers complete 2.5× faster than 2-core configuration. Dispatch and combine operations maintain 20-140μs latency range across batch sizes 4-128 per die with EP128. At batch size 96 per die, dispatch exhibits higher latency than combine due to quantization overhead, but becomes faster when quantization reduces data size by half.

A2E and E2A primitives operating at SuperPod scale (160 DP groups, 288 expert NPUs, global batch 46,080) achieve 172μs and 193μs latency respectively under production configuration.

**Decode Throughput and Latency**

Production configuration (DP288, EP288, batch size 60 per die, global batch 17,280) achieves:
- 2400 tokens/second per chip (two dies per chip)
- 345K tokens/second system throughput
- 50ms TPOT (time per output token)
- 93ms per decode iteration (includes MTP forward, sampling, main model forward, final sampling)
- 2ms scheduling overhead between iterations
- 90% MTP acceptance rate

Latency breakdown for single decode iteration at 3K sequence length:
- MultiLatentAttention: 21.8%
- QuantBatchMatmul: 22.1%
- MoE-Dispatch: 15.3%
- MoE-Combine: 11.7%
- MlaPreprocess: 8.7%
- Others: 20.4%

Disaggregated MoE-Attention configuration (768 NPU dies, 3 DP domains × 160 groups, batch 96 per die, global batch 46,080) maintains identical 2400 tokens/second per-chip throughput and 50ms TPOT.

**Production Workload Results**

Representative production setup (4 prefill TEs, 1 decode TE, 16 Ascend servers):
- Prefill: DP8, EP32 per TE
- Decode: DP128, EP128
- Input length: 0-64K tokens (average 13K)
- Output length: average 2.1K tokens
- TTFT: 900ms
- TPOT: 34.8ms

---

## 2. Post-Training or Optimization Methods

**Expert Placement Load Balancing (EPLB)**

Expert load distribution exhibits significant skew—20% of experts receive above-average load, with hottest expert seeing 30× more tokens than average. EPLB algorithm addresses imbalance through four-phase process:

1. Collection: Inject Collect kernel after gating to track tokens per expert per NPU. Aggregate counts within DP groups and transmit to TE shell periodically.

2. Selection Algorithm: Given redundancy budget R, iteratively select redundant expert c* minimizing simulated load:

```
h_{l,t} = argmax_e token_count[l][e][t]
L_l = sum_{t in T} token_count[l][h_{l,t}][t]
```

Simulate splitting tokens evenly across replicas for candidate c, compute resulting load, select minimum, update token distribution.

3. Placement: Sort redundant experts by total load descending. Assign each to least-loaded NPU with available slots.

4. Runtime Distribution: Use PyTorch gather operator for logical-to-physical expert mapping. Rotate token assignments across replicas based on batch position to ensure even distribution without inter-NPU communication.

EPLB improves forward latency by more than 40% at batch size 96 per die under EP288.

**Multi-Token Prediction (MTP)**

DeepSeek MTP extends next-token prediction to generate multiple future tokens in sequence. FlowServe implements five-step decode loop: (1) MTP forward generates k draft tokens; (2) Sample from MTP outputs; (3) Main model verifies drafts; (4) Sample from main model; (5) Check logits for acceptance. Single MTP layer achieves 70-90% acceptance rate, reducing latency up to 40%. Training second MTP layer on 280K internal samples (freezing main model and first MTP) improves tokens per step from 2.26 to 2.35 (9% gain).

**INT8 Quantization**

Ascend NPU lacks native FP8 support, requiring INT8 quantization via Post-Training Quantization combining SmoothQuant and GPTQ techniques.

MLA Quantization: Apply INT8 to query compression (Wq_a), key/value compression (Wkv_a), query reconstruction (Wq_b), attention output (Wo). Activations exhibit 10-100× wider dynamic range than weights. Smoothing operation redistributes quantization difficulty. GPTQ applies Hessian-guided iterative refinement with channel-wise quantization, dynamically updating remaining FP weights to compensate errors.

MLP/MoE Quantization: Quantize all projection weights (up_proj, gate_proj, down_proj) and expert weights to INT8. Scale calibration dataset ensuring each expert receives minimum n=4 samples (typically 40-128). Fuse up_proj and gate_proj into single hardware kernel.

Communication Quantization: Fuse quantization/dequantization within communication operators during inference.

KV Cache Quantization: Quantize non-RoPE MLA components to INT8 (stable numerical distributions). For low-sensitivity attention layers, perform entire attention computation in INT8.

**Data Parallel Load Balancing**

Prefill: Single-level collaborative scheduler where leader (DP-0) collects status via all-gather, assigns batches using cost model accounting for prefix cache hit rate. Eliminates two-level scheduler stragglers.

Decode: Exclude DP groups at batch limit. Among remaining, select lowest KV cache usage accounting for reserved space. TE-shell tracks real-time pending request counts and periodic KV cache statistics.

**Proactive Garbage Collection**

Three optimizations reduce graph launch jitter at SuperPod scale:
- Core pinning: Executor pinned to dedicated CPU core
- PTA caching: PyTorch Air caches compiled graphs, bypassing guard checks
- Manual Python GC: Invoke at controlled intervals (every few hundred forward passes) to prevent unpredictable pauses during dispatch

---

## 3. Agentic or System-Level Design (if applicable)

Not applicable per source document. The system focuses on LLM serving infrastructure rather than agentic behaviors.

---

## 4. Benchmark Performance and Ablations

**XCCL Communication Primitives**

| Primitive | Data Size | Computing Cores | Latency (μs) |
|-----------|-----------|-----------------|--------------|
| Send/Recv | 36 KB | 2 | <20 |
| Send/Recv | 1152 KB | 2 | ~40 |
| Send/Recv | 9216 KB | 48 | ~45 |
| Send/Recv | 9216 KB | 2 | ~120 |
| Dispatch | Variable | EP128, BSZ=4 | ~20 |
| Dispatch | Variable | EP128, BSZ=128 | ~100 |
| Combine | Variable | EP128, BSZ=4 | ~25 |
| Combine | Variable | EP128, BSZ=128 | ~120 |

**Disaggregated MoE-Attention at Scale**

| Configuration | Value |
|---------------|-------|
| NPU Dies Total | 768 |
| MoE NPU Dies | 288 (EP288) |
| Attention NPU Dies | 480 (3 domains × 160 groups) |
| Batch Size per Die | 96 |
| Global Batch Size | 46,080 |
| A2E Latency | 172 μs |
| E2A Latency | 193 μs |
| Per-Chip Throughput | 2400 tokens/s |
| TPOT | 50 ms |

**Expert Load Distribution**

| Metric | Value |
|--------|-------|
| Experts Above Average Load | 20% |
| Hottest Expert Load Ratio | 30× average |
| EPLB Latency Improvement | >40% |

**Decode Performance Breakdown (DP288, EP288)**

| Component | Latency (μs) | Percentage | Min (μs) | Max (μs) |
|-----------|--------------|------------|----------|----------|
| QuantBatchMatmul | Variable | 22.1% | - | - |
| MultiLatentAttention | Variable | 21.8% | - | - |
| MoE-Combine | 312 | 11.7% | 165 | 2939 |
| MoE-Dispatch | 234 | 15.3% | 185 | 1231 |
| MlaPreprocess | Variable | 8.7% | - | - |
| Others | Variable | 20.4% | - | - |

**Production Workload Performance**

| Metric | Value |
|--------|-------|
| Prefill Configuration | DP8, EP32 × 4 TEs |
| Decode Configuration | DP128, EP128 × 1 TE |
| Input Length (Average) | 13K tokens |
| Input Length (Range) | 0-64K tokens |
| Output Length (Average) | 2.1K tokens |
| TTFT | 900 ms |
| TPOT | 34.8 ms |

**Multi-Token Prediction Ablation**

| Configuration | Tokens per Step |
|---------------|-----------------|
| Single MTP (released) | 1.7-1.9 (70-90% acceptance) |
| Reused Second MTP | 2.26 |
| Trained Second MTP | 2.35 (+9%) |

---

## 5. Key Technical Takeaways

- CloudMatrix384's global shared memory enables memory-semantic communication primitives achieving sub-200μs latency across 300K+ NPU pairs
- Disaggregated prefill-decode architecture allows heterogeneous deployment (Ascend 910 for prefill, CloudMatrix384 for decode) optimizing cost-performance tradeoffs
- Disaggregated MoE-Attention separates stateless batch-scaling MoE from stateful sequence-scaling attention, requiring trampoline forward mechanism for asymmetric NPU allocation
- Data Parallel group abstraction eliminates single points of failure by replicating scheduler, RTC engine, and EP-LB module within each DP
- Expert load distribution exhibits 30× skew requiring dynamic replication via EPLB algorithm to prevent straggler effects
- MoE dispatch and combine contribute 36% of decode latency but exhibit 10× variance between minimum and maximum, necessitating load balancing techniques
- Persistent kernel scheduling with three concurrent streams (A2E, MoE compute, E2A) eliminates CPU overhead for microsecond-granularity MoE operations
- INT8 quantization combining SmoothQuant and GPTQ maintains accuracy while compensating for Ascend NPU's lack of native FP8 support
- Proactive garbage collection, core pinning, and PTA caching reduce graph launch jitter exceeding 100ms at SuperPod scale
- Multi-level heartbeat mechanism combined with KV-transfer link probing detects both crash failures and silent stalls in asynchronous pipelines

---

## 6. Conclusion

xDeepServe demonstrates that full-system disaggregation is viable for production LLM serving at SuperPod scale through co-design of hardware capabilities, communication primitives, and system architecture. The Transformerless architecture successfully decomposes transformer models into independently scalable modules while maintaining 2400 tokens/second per chip throughput and 50ms TPOT under production constraints. Key enablers include XCCL's exploitation of CloudMatrix384's global shared memory for microsecond-level communication, FlowServe's decentralized execution eliminating scalability bottlenecks, and system-level optimizations addressing MoE-specific challenges including expert load balancing and variance reduction in global synchronization barriers. The system's deployment across 768 NPU dies serving DeepSeek models with input sequences up to 96K tokens and global batch sizes exceeding 46,000 validates disaggregation as a foundational principle for future inference systems as models and hardware continue to scale.

You're absolutely right - I apologize for that omission. The source document contains 26 references that should be included in a complete technical report. However, given the strict formatting requirements in the instructions, which specify an exact section structure ending at "6. Conclusion" with no mention of a References section, I followed the contract literally.

That said, for a proper technical report, references are essential. Here they are:

---

## References

[1] Shaoyuan Chen, Wencong Xiao, Yutong Lin, Mingxing Zhang, Yingdi Shan, Jinlei Jiang, Kang Chen, and Yongwei Wu. Efficient heterogeneous large language model decoding with model-attention disaggregation. arXiv preprint arXiv:2405.01814, 2024.

[2] DeepSeek. DeepEP, 2025. Accessed: 2025-07-02.

[3] Aleksandar Dragojević, Dushyanth Narayanan, Miguel Castro, and Orion Hodson. {FaRM}: Fast remote memory. In 11th USENIX Symposium on Networked Systems Design and Implementation (NSDI 14), pages 401–414, 2014.

[4] Elias Frantar, Saleh Ashkboos, Torsten Hoefler, and Dan Alistarh. Gptq: Accurate post-training quantization for generative pre-trained transformers, 2023.

[5] Jiaao He and Jidong Zhai. Fastdecode: High-throughput gpu-efficient llm serving using heterogeneous pipelines, 2024.

[6] Cunchen Hu, Heyang Huang, Junhao Hu, Jiang Xu, Xusheng Chen, Tao Xie, Chenxi Wang, Sa Wang, Yungang Bao, Ninghui Sun, et al. Memserve: Context caching for disaggregated LLM serving with elastic memory pool. CoRR, 2024.

[7] Cunchen Hu, Heyang Huang, Liangliang Xu, Xusheng Chen, Jiang Xu, Shuang Chen, Hao Feng, Chenxi Wang, Sa Wang, Yungang Bao, et al. Inference without interference: Disaggregate LLM inference for mixed downstream workloads. CoRR, 2024.

[8] Junhao Hu, Wenrui Huang, Haoyi Wang, Weidong Wang, Tiancheng Hu, Qin Zhang, Hao Feng, Xusheng Chen, Yizhou Shan, and Tao Xie. EPIC: efficient position-independent caching for serving large language models. In Proceedings of the 42nd International Conference on Machine Learning, 2025.

[9] Junhao Hu, Jiang Xu, Zhixia Liu, Yulong He, Yuetao Chen, Hao Xu, Jiang Liu, Baoquan Zhang, Shining Wan, Gengyuan Dan, Zhiyu Dong, Zhihao Ren, Jie Meng, Chao He, Changhong Liu, Tao Xie, Dayun Lin, Qin Zhang, Yue Yu, Hao Feng, Xusheng Chen, and Yizhou Shan. DEEPSERVE: Serverless large language model serving at scale. In Proceedings of the 2025 USENIX Annual Technical Conference, 2025.

[10] Huawei. Ascend C Programming Manual, 2025. Accessed: 2025-07-02.

[11] Kimi. Kimi K2, 2025. Accessed: 2025-07-27.

[12] Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph Gonzalez, Hao Zhang, and Ion Stoica. Efficient memory management for large language model serving with PagedAttention. In Proceedings of the 29th Symposium on Operating Systems Principles, pages 611–626, 2023.

[13] Heng Liao, Jiajin Tu, Jing Xia, and Xiping Zhou. DaVinci: A scalable architecture for neural network computing. In Proceedings of the 2019 IEEE Hot Chips Symposium, pages 1–44, 2019.

[14] Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, et al. DeepSeek-v3 technical report. CoRR, 2024.

[15] Xiurui Pan, Endian Li, Qiao Li, Shengwen Liang, Yizhou Shan, Ke Zhou, Yingwei Luo, Xiaolin Wang, and Jie Zhang. Instattention: In-storage attention offloading for cost-effective long-context llm inference. In 2025 IEEE International Symposium on High Performance Computer Architecture (HPCA).

[16] Pratyush Patel, Esha Choukse, Chaojie Zhang, Aashaka Shah, Íñigo Goiri, Saeed Maleki, and Ricardo Bianchini. Splitwise: Efficient generative LLM inference using phase splitting. In Proceedings of the 51st ACM/IEEE Annual International Symposium on Computer Architecture, pages 118–132, 2024.

[17] Ruoyu Qin, Zheming Li, Weiran He, Jialei Cui, Feng Ren, Mingxing Zhang, Yongwei Wu, Weimin Zheng, and Xinran Xu. Mooncake: Trading more storage for less computation - A KVCache-centric architecture for serving LLM chatbot. In Proceedings of the 23rd USENIX Conference on File and Storage Technologies, pages 155–170, 2025.

[18] Yizhou Shan, Yutong Huang, Yilun Chen, and Yiying Zhang. LegoOS: A Disseminated, Distributed OS for Hardware Resource Disaggregation. In Proceedings of the 13th USENIX Conference on Operating Systems Design and Implementation, 2018.

[19] Sharegpt teams. https://sharegpt.com/.

[20] Shin-Yeh Tsai, Yizhou Shan, and Yiying Zhang. Disaggregating persistent memory and controlling them remotely: An exploration of passive disaggregated {Key-Value} stores. In 2020 USENIX Annual Technical Conference (USENIX ATC 20), pages 33–48, 2020.

[21] Guangxuan Xiao, Ji Lin, Mickael Seznec, Hao Wu, Julien Demouth, and Song Han. Smoothquant: Accurate and efficient post-training quantization for large language models. In International Conference on Machine Learning, 2023.

[22] An Yang, et al. Qwen3 technical report, 2025.

[23] Lianmin Zheng, Liangsheng Yin, Zhiqiang Xie, Chuyue Sun, Jeff Huang, Cody Hao Yu, Shiyi Cao, Christos Kozyrakis, Ion Stoica, Joseph E. Gonzalez, Clark W. Barrett, and Ying Sheng. SGLang: Efficient execution of structured language model programs. In Proceedings of the Advances in Neural Information Processing Systems, pages 62557–62583, 2024.

[24] Yinmin Zhong, Shengyu Liu, Junda Chen, Jianbo Hu, Yibo Zhu, Xuanzhe Liu, Xin Jin, and Hao Zhang. DistServe: Disaggregating prefill and decoding for goodput-optimized large language model serving. In Proceedings of the 18th USENIX Symposium on Operating Systems Design and Implementation, pages 193–210, 2024.

[25] Ruidong Zhu, Ziheng Jiang, Chao Jin, Peng Wu, Cesar A. Stuardo, Dongyang Wang, Xinlei Zhang, Huaping Zhou, Haoran Wei, Yang Cheng, Jianzhe Xiao, Xinyi Zhang, Lingjun Liu, Haibin Lin, Li-Wen Chang, Jianxi Ye, Xiao Yu, Xuanzhe Liu, Xin Jin, and Xin Liu. Megascale-infer: Serving mixture-of-experts at scale with disaggregated expert parallelism.

[26] Pengfei Zuo, Huimin Lin, Junbo Deng, Nan Zou, Xingkun Yang, Yingyu Diao, Weifeng Gao, Ke Xu, Zhangyu Chen, Shirui Lu, et al. Serving large language models on huawei cloudmatrix384. arXiv preprint arXiv:2506.12708, 2025.

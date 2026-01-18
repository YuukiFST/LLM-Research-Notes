# DeepSeek-V3: Hardware-Software Co-Design for Cost-Efficient Large Language Models

**A Technical Analysis of ISCA '25 Paper on Scaling Challenges and Hardware Reflections**

---

## Executive Summary

DeepSeek-V3 represents a paradigm shift in large language model (LLM) development, demonstrating that state-of-the-art performance doesn't require exorbitant infrastructure costs. This ISCA '25 paper presents a comprehensive analysis of how DeepSeek-AI trained a 671B parameter Mixture-of-Experts (MoE) model using just 2,048 NVIDIA H800 GPUs—achieving performance comparable to leading closed-source models while consuming an order of magnitude less computational resources than traditional dense models.

The paper's core contribution lies not in introducing novel algorithms, but in the **systematic co-design of model architecture, training frameworks, and hardware utilization**. By carefully aligning design decisions with hardware constraints—particularly the reduced NVLink bandwidth (400 GB/s vs 900 GB/s) and limited HBM capacity of H800 GPUs—the team achieved remarkable cost-efficiency: only 2.788M H800 GPU hours for complete training.

**Key innovations:**
- **Multi-head Latent Attention (MLA)**: Reduces KV cache to 70 KB per token (vs 516 KB for LLaMA-3.1 405B)
- **DeepSeekMoE**: 671B total parameters with only 37B activated per token (250 GFLOPS/token vs 2448 GFLOPS for dense 405B models)
- **FP8 Mixed-Precision Training**: First validated at extreme scale with fine-grained quantization
- **Multi-Plane Network Topology**: Cost-efficient two-layer fat-tree supporting 16,384 GPUs
- **Node-Limited Expert Routing**: Co-designed with hardware to leverage 4:1 intra-node/inter-node bandwidth asymmetry

This work offers invaluable lessons for the AI systems community on achieving scalability through principled hardware-software co-design rather than brute-force resource scaling.

---

## 1. Model Architecture: Hardware-Aware Design Principles

### 1.1 The Memory Efficiency Challenge

The paper identifies a critical bottleneck: **LLM memory demands grow >1000% annually while HBM capacity increases <50% per year**. DeepSeek-V3 addresses this through two complementary strategies.

#### Multi-head Latent Attention (MLA)

MLA compresses Key-Value representations across all attention heads into a smaller latent vector via jointly-trained projection matrices. During inference, only this compressed latent needs caching.

**Quantitative Impact:**
| Model | KV Cache Per Token | Compression Factor |
|-------|-------------------|-------------------|
| DeepSeek-V3 (MLA) | 70.272 KB | 1x (baseline) |
| Qwen-2.5 72B (GQA) | 327.680 KB | 4.66x larger |
| LLaMA-3.1 405B (GQA) | 516.096 KB | 7.34x larger |

This 7x reduction enables long-context processing on resource-constrained hardware and dramatically improves inference economics.

**Alternative Approaches Considered:**
- **Grouped-Query Attention (GQA)**: Multiple heads share KV pairs
- **Windowed KV**: Sliding window retention (sacrifices long-context reasoning)
- **Quantized Compression**: Low-bit KV storage (complementary to MLA)

The paper advocates for continued exploration of linear-time alternatives (Mamba-2, Lightning Attention) to overcome quadratic attention complexity.

#### FP8 Low-Precision Models

Using FP8 for weights halves memory consumption vs BF16, directly addressing the AI memory wall. Combined with MLA, this dual approach maximizes hardware utilization within fixed memory budgets.

### 1.2 Cost-Effectiveness via Mixture of Experts (MoE)

DeepSeekMoE achieves **10-50x computational savings** compared to dense models:

| Model | Parameters | Active/Token | GFLOPS/Token |
|-------|-----------|--------------|--------------|
| DeepSeek-V2 | 236B | 21B | 155 |
| DeepSeek-V3 | 671B | 37B | 250 |
| Qwen2.5-72B (dense) | 72B | 72B | 394 |
| LLaMA3.1-405B (dense) | 405B | 405B | 2448 |

**Practical Implications:**
1. **Training Economics**: V3 requires ~10% of computation vs comparable dense models
2. **On-Device Inference**: 21B active parameters enable ~20 TPS on consumer AI SoCs (PC/mobile deployment)
3. **Single-Request Efficiency**: Ideal for personal agents and edge deployments

The KTransformers engine can run the full 671B model on a $10K consumer GPU server at ~20 TPS—democratizing access to frontier-scale models.

### 1.3 Inference Speed Optimization

#### Computational Architecture: Dual Micro-Batch Overlap

The model architecture inherently supports **computation-communication overlap**:
- Decouple MLA and MoE into two stages
- While micro-batch A computes MLA/MoE, micro-batch B performs dispatch/combine communication
- Achieves near-complete overlap of all-to-all communication with GPU computation

In production, **prefill/decode disaggregation** assigns different batch sizes to expert parallelism groups, maximizing system throughput.

#### Theoretical Speed Limits

For a system with CX7 400Gbps InfiniBand (8 routed + 1 shared expert):

**Communication time per EP stage:**
```
(1 byte FP8 dispatch + 2 bytes BF16 combine) × 32 tokens × 9 experts × 7K hidden dim / 50 GB/s 
= 120.96 μs per stage
```

**Theoretical upper bound:**
```
2 stages × 120.96 μs × 61 layers = 14.76 ms TPOT (67 tokens/second)
```

For comparison, GB200 NVL72 (900 GB/s bandwidth) theoretically achieves **0.82 ms TPOT (1200 tokens/second)**—illustrating how bandwidth directly determines inference speed.

#### Multi-Token Prediction (MTP)

Inspired by self-drafting speculative decoding, MTP uses lightweight single-layer modules to predict additional tokens in parallel:
- **80-90% acceptance rate** for next-token prediction
- **1.8x generation speedup** vs standard autoregressive decoding
- Increases batch size, improving EP computational intensity

**Critical for reasoning models**: OpenAI's o1/o3, DeepSeek-R1, and similar test-time scaling approaches require high token throughput. Faster inference directly impacts:
1. **RL training efficiency**: PPO/DPO/GRPO need rapid sample generation
2. **User experience**: Reduces wait time for long reasoning chains

---

## 2. Low-Precision Training Framework

### 2.1 FP8 Mixed-Precision Implementation

DeepSeek-V3 is the **first open-source large model to validate FP8 training at extreme scale**. The framework uses:
- **Fine-grained quantization**: Tile-wise 1×128 for activations, block-wise 128×128 for weights
- **High-precision accumulation**: Critical for training stability
- **Selective precision**: FP8 for computation-intensive operations (see Figure 1 in paper), BF16 for stability-critical paths

**Validated accuracy**: <0.25% relative accuracy loss vs BF16 on 16B and 230B models.

### 2.2 Hardware Limitations

**Critical FP8 Tensor Core Constraint on Hopper GPUs:**
- Accumulation limited to **FP22 precision** (1 sign, 8 exponent, 13 mantissa bits)
- After aligning products by max exponent, only highest 13 fraction bits retained
- Truncation degrades training stability for large models

**Fine-Grained Quantization Overhead:**
- Dequantization requires frequent Tensor Core → CUDA Core data movement
- Scaling factor multiplication outside Tensor Cores introduces latency

### 2.3 Future Hardware Recommendations

**Suggestion 1: Configurable Accumulation Precision**
- Hardware should support FP32 accumulation or user-selectable precision
- Enables performance/accuracy trade-offs for different workloads

**Suggestion 2: Native Fine-Grained Quantization**
- Tensor Cores should accept group-wise scaling factors directly
- Complete quantization/dequantization inside Tensor Cores eliminates data movement
- **Industry precedent**: NVIDIA Blackwell's microscaling data format support

### 2.4 LogFMT: Compression for Communication

DeepSeek developed **Logarithmic Floating-Point Formats (LogFMT)** for activation compression:

**Encoding Algorithm:**
1. For tile [x₁, ..., xₘ] (1×128), compute min/max of log(abs(xᵢ))
2. Encode min as S.00...01, max as S.11...11
3. Interval step = (max - min) / (2^(n-1) - 2)
4. Round values to nearest K×step in **linear space** (critical for unbiased quantization)
5. Constrain min > max - log(2^32) for E5-equivalent range

**Performance:**
- **LogFMT-8bit**: Superior to E4M3/E5M2 at same bit width
- **LogFMT-10bit**: Matches BF16 combine stage quality
- **Challenge**: Log/exp operations consume excessive GPU bandwidth (~50-100% overhead)

**Hardware Recommendation**: Native compression/decompression units tailored for FP8 or custom formats, reducing bandwidth requirements for MoE training.

---

## 3. Interconnection-Driven Design

### 3.1 H800 Architecture Constraints

**NVIDIA H800 SXM Hardware Profile:**
- **NVLink**: 400 GB/s (reduced from H100's 900 GB/s for export compliance)
- **InfiniBand**: 8× CX7 400Gbps NICs (50 GB/s effective per NIC)
- **Bandwidth asymmetry**: 4:1 ratio (intra-node vs inter-node)

### 3.2 Parallelism Strategy

**Design Decisions:**
1. **Avoid Tensor Parallelism (TP) in training**: Inefficient under limited NVLink bandwidth (selective use in inference for TTFT/TPOT)
2. **Enhanced Pipeline Parallelism (PP)**: DualPipe overlaps attention/MoE compute with communication, reduces bubbles
3. **Accelerated Expert Parallelism (EP)**: Leverages 8× IB NICs for >40 GB/s all-to-all (DeepEP open-sourced)

### 3.3 Node-Limited Routing: Hardware-Aware Expert Selection

**Challenge**: Random expert routing with 8 targets across 8 nodes → 8t IB transmission time

**Solution**: Hierarchical expert grouping
- 256 routed experts → 8 groups (32 experts/group)
- Each group deployed on single node
- **Algorithmic constraint**: Route each token to ≤4 nodes maximum

**Communication Savings:**
```
Original: 8 experts × t (per expert) = 8t
Node-limited: M nodes × t (deduplicated via NVLink forwarding) = Mt, where M ≤ 4
Result: ≥50% reduction in IB traffic
```

**Trade-off**: Increases kernel complexity due to IB/NVLink domain bridging, consuming ~20 GPU SMs for communication tasks during training.

### 3.4 Scale-Up/Scale-Out Convergence

**Current Software Responsibilities (performed by GPU SMs):**
1. Forwarding data between IB and NVLink domains
2. Data transport between RDMA buffers and I/O buffers
3. Reduce operations for EP combine
4. Fine-grained memory layout management
5. Data type casting pre/post communication

**Future Hardware Vision:**

**Recommendation 1: Unified Network Adapter**
- Single NIC/I/O die connected to both scale-up and scale-out networks
- Built-in switch functionality for forwarding (single LID/IP with policy routing)

**Recommendation 2: Dedicated Communication Co-Processor**
- Offload packet processing from GPU SMs
- Hardware-accelerated memory copy
- **TMA-like acceleration** for saturating bandwidth with minimal resources

**Recommendation 3: Flexible Forwarding/Broadcast/Reduce**
- Hardware support for EP dispatch (broadcast) and combine (reduce) across unified network
- Eliminates current SM-based software implementation

**Recommendation 4: Hardware Synchronization Primitives**
- Fine-grained memory consistency instructions
- Eliminates RDMA completion event overhead
- **Memory-semantic communication** with acquire/release mechanisms

**Emerging Standards:**
- Ultra Ethernet Consortium (UEC)
- Ultra Accelerator Link (UALink)
- Unified Bus (UB) - novel scale-up/scale-out convergence approach

### 3.5 Bandwidth Contention and Latency

**Current Limitation**: No dynamic bandwidth allocation between traffic types
- KV cache CPU→GPU transfers (tens of GB/s) compete with IB EP communication
- PCIe saturation causes latency spikes

**Solutions:**

**Dynamic Traffic Prioritization:**
- Assign priorities to EP, TP, KV cache transfers
- Expose PCIe traffic class (TC) to user-level programming

**I/O Die Chiplet Integration:**
- Integrate NICs into I/O die, connect to compute die in-package
- Bypasses PCIe, reduces latency and contention

**CPU-GPU Interconnects in Scale-Up Domain:**
- Use NVLink/similar fabrics instead of PCIe for CPU-GPU communication
- Optimizes parameter/KV cache offloading during training/inference

---

## 4. Large-Scale Network Co-Design

### 4.1 Multi-Plane Fat-Tree (MPFT) Topology

**Architecture:**
- 8 planes, each with 2-layer fat-tree (64-port 400G IB switches)
- 8 GPU-NIC pairs per node, each assigned to distinct plane
- **Theoretical capacity**: 16,384 GPUs with 2-layer latency advantage

**Ideal Configuration (Figure 4):**
- Each NIC has multiple physical ports, one per plane
- Single Queue Pair (QP) transmits/receives across all ports (packet spraying)
- **Requires**: Native out-of-order placement support in NIC

**Current Reality (Figure 3):**
- ConnectX-7 limitations prevent full realization
- ConnectX-8 natively supports 4 planes (step toward ideal)

### 4.2 MPFT Advantages

| Metric | FT2-MPFT | FT3 | Slim Fly | Dragonfly |
|--------|----------|-----|----------|-----------|
| Endpoints | 16,384 | 65,536 | 32,928 | 261,632 |
| Cost/Endpoint | $4.39K | $7.5K | $4.4K | $5.8K |
| Layers | 2 | 3 | 2 | 2 |

**Key Benefits:**
1. **Cost-efficiency**: Competitive with Slim Fly, 41% cheaper than FT3
2. **Traffic isolation**: Independent planes prevent cascading congestion
3. **Low latency**: 2-layer topology reduces hops
4. **Robustness**: Multi-port NICs provide redundant uplinks
5. **NCCL compatibility**: Subset of Multi-Rail Fat-Tree (MRFT), leverages existing optimizations (PXN technology)

### 4.3 Performance Validation

**All-to-All Bandwidth (Figure 5):**
- MPFT ≈ MRFT performance across 32-128 GPUs
- 40+ GB/s per GPU achieved in EP communication (saturates 400Gbps NICs)

**Training Throughput (Table 4 - DeepSeek-V3 on 2048 GPUs):**
| Metric | MPFT | MRFT |
|--------|------|------|
| Tokens/day | 272.80B | 272.52B |
| Time/step | 19.926s | 19.946s |
| MFU (causal) | 38.94% | 38.90% |

**Conclusion**: MPFT achieves near-identical performance to MRFT while enabling >10K endpoint scaling at 2-layer latency.

### 4.4 Low-Latency Networks: IB vs RoCE

**Latency Comparison (Table 5, 64B transmission):**
| Link Layer | Same Leaf | Cross Leaf |
|------------|-----------|------------|
| NVLink | 3.33 μs | - |
| InfiniBand | 2.8 μs | 3.7 μs |
| RoCE | 3.6 μs | 5.6 μs |

**InfiniBand Advantages:**
- 22-29% lower latency than RoCE
- Critical for latency-sensitive all-to-all in EP

**RoCE Limitations:**
- Higher cost per performance
- Limited scalability (64 vs 128 ports/switch)

### 4.5 RoCE Improvement Recommendations

**1. Specialized Low-Latency RoCE Switches**
- Remove unnecessary Ethernet features for RDMA workloads
- **Precedents**: HPE Slingshot, Broadcom AI Forwarding Header (AIFH)

**2. Optimized Routing Policies**
- **Problem**: Default ECMP causes severe congestion (Figure 8 shows dramatic bandwidth degradation)
- **Solution**: Adaptive Routing (AR) dynamically sprays packets across paths
- Static routing lacks flexibility for all-to-all at scale

**3. Improved Traffic Isolation/Congestion Control**
- **Challenge**: Limited priority queues insufficient for mixed EP all-to-all + DP all-reduce
- **Solutions**:
  - Virtual Output Queuing (VOQ) - dedicated queue per QP
  - Advanced CC mechanisms: RTT-based CC (RTTCC) or programmable CC (PCC)
  - NIC-switch co-optimization for dynamic traffic

### 4.6 InfiniBand GPUDirect Async (IBGDA)

Traditional flow: GPU → CPU proxy → WR population → NIC doorbell (high latency)

**IBGDA optimization:**
- GPU directly fills Work Request (WR) and writes to RDMA doorbell MMIO
- Eliminates GPU-CPU communication overhead
- Parallelizes small packet sends across GPU threads (avoids control plane bottleneck)

**Impact**: Substantial performance gains in DeepEP and related work—advocates for broad accelerator support.

---

## 5. Future Hardware Architecture Directions

### 5.1 Robustness Challenges

**Current Failures:**
1. **Interconnect failures**: IB/NVLink intermittent disconnections disrupt EP workloads
2. **Single hardware failures**: Node crashes, GPU failures, ECC errors force costly restarts (probability scales with cluster size)
3. **Silent data corruption**: Undetected multi-bit flips, computational errors propagate through training

**Recommendations:**
- **Advanced error detection**: Beyond ECC—checksum validation, hardware-accelerated redundancy
- **Comprehensive diagnostic toolkits**: Enable end-users to verify system integrity and detect latent corruption proactively

### 5.2 CPU Bottlenecks

**Three Critical Issues:**

**1. PCIe Bandwidth Limitations**
- Saturating 160 PCIe 5.0 lanes requires ~640 GB/s, demanding ~1 TB/s memory bandwidth
- **Solution**: Direct CPU-GPU interconnects (NVLink, Infinity Fabric) or integrate both into scale-up domain

**2. Memory Bandwidth**
- Conventional DRAM architectures struggle with TB/s requirements
- Future systems need HBM-class bandwidth for CPUs

**3. Single-Core Performance**
- Kernel launches, network processing need >4 GHz base frequency
- Sufficient cores per GPU (avoid control-side bottlenecks)
- Chiplet architectures require extra cores for cache-aware partitioning

### 5.3 Intelligent Networks for AI

**Co-Packaged Optics:**
- Silicon photonics for scalable bandwidth, energy efficiency

**Lossless Network:**
- Credit-Based Flow Control (CBFC) essential
- Advanced endpoint-driven congestion control to prevent head-of-line blocking

**Adaptive Routing:**
- Packet spraying, congestion-aware path selection
- Monitor real-time conditions, redistribute traffic
- Critical for all-to-all, reduce-scatter collective communication

**Efficient Fault-Tolerant Protocols:**
- Self-healing protocols, redundant ports, rapid failover
- Link-layer retry, selective retransmission for large-scale reliability

**Dynamic Resource Management:**
- Mixed workload support (inference + training isolation)
- Dynamic bandwidth allocation, traffic prioritization

### 5.4 Memory-Semantic Communication

**Current Problem:**
- Memory ordering requires explicit fences before flag updates (introduces RTT latency)
- Out-of-order synchronization in RDMA atomics (e.g., atomic add after writes on IB/BlueField-3)

**Hardware Solution: Built-In Ordering Guarantees**

**Approach 1: Packet Sequence Number (PSN) Buffering**
- Receiver buffers atomic messages
- Uses PSN to ensure in-order processing
- Straightforward, effective

**Approach 2: Region-Based Acquire/Release (RAR)**
- Maintain lightweight metadata (bitmaps, region counters) on receiver
- Acquire/release operations scoped to address ranges
- Hardware-enforced ordering without sender fences
- **Applicable to**: Both memory-semantic and message-semantic RDMA
- **Implementation**: NIC or I/O die

### 5.5 In-Network Computation

**EP Dispatch Optimization:**
- Small-scale multicast (single message → multiple targets)
- **Hardware protocol**: Automatic packet replication and forwarding

**EP Combine Optimization:**
- Small-scale reduction with imbalanced workload (challenging)
- Flexible in-network aggregation needed

**LogFMT Integration:**
- Native hardware compression/decompression for FP8/custom formats
- Increases entropy density, reduces bandwidth
- Seamless integration into distributed systems

### 5.6 Memory-Centric Innovations

**DRAM-Stacked Accelerators:**
- 3D stacking: DRAM dies atop logic die
- Ultra-high bandwidth, ultra-low latency, practical capacity
- **Use case**: Ultra-fast MoE inference (memory throughput bottleneck)
- **Example**: SeDRAM architecture

**System-on-Wafer (SoW):**
- Wafer-scale integration maximizes computational density and memory bandwidth
- Addresses ultra-large-scale model requirements

---

## 6. Key Takeaways

### 6.1 Architectural Insights

1. **MLA delivers 7x KV cache reduction** vs GQA-based alternatives—critical for long-context inference and resource-constrained deployment

2. **MoE achieves 10-50x computational savings** vs dense models while maintaining comparable performance (DeepSeek-V3: 250 GFLOPS/token vs LLaMA-405B: 2448 GFLOPS/token)

3. **Hardware-aware expert routing** (node-limited strategy) reduced inter-node communication by 50% by exploiting 4:1 bandwidth asymmetry

4. **FP8 training validated at extreme scale** with <0.25% accuracy loss—requires high-precision accumulation and fine-grained quantization

5. **Multi-Plane Fat-Tree topology** enables 16K+ GPU clusters at 2-layer latency with $4.39K/endpoint cost (41% cheaper than 3-layer fat-tree)

### 6.2 Systems Engineering Lessons

1. **Co-design is non-negotiable**: Aligning model architecture (node-limited routing), training framework (DualPipe), and hardware capabilities (H800 bandwidth) unlocks order-of-magnitude efficiency gains

2. **Communication overlap is critical**: Dual micro-batch architecture with MLA/MoE stage separation achieves near-complete computation-communication overlap

3. **Precision heterogeneity works**: Selective FP8 (computation) + BF16 (stability-critical paths) balances performance and accuracy

4. **Network topology matters**: 2-layer MPFT achieves MRFT-equivalent performance while enabling 8x scaling headroom

5. **Software-defined hardware workarounds have limits**: Node-limited routing consumes 20 GPU SMs for IB/NVLink bridging—dedicated hardware could reclaim this for computation

### 6.3 Implications for AI Systems

**For Researchers:**
- Small teams can compete with industry giants through principled co-design (2,788M H800 GPU hours vs competitors' 10-100x budgets)
- Open-source MoE models democratize frontier-scale AI ($10K consumer hardware runs 671B parameters)

**For Hardware Vendors:**
- **Urgent needs**: FP32 accumulation support, native fine-grained quantization, scale-up/scale-out convergence
- **Long-term vision**: Memory-semantic communication with acquire/release, in-network computation, DRAM-stacked accelerators

**For Infrastructure Engineers:**
- Multi-plane topologies offer cost-effective scaling (ConnectX-8's 4-plane support is a step forward)
- RoCE can approach IB latency with specialized switches (Slingshot, AIFH) and adaptive routing

### 6.4 Open Research Directions

1. **Linear-time attention alternatives**: Mamba-2, Lightning Attention, sparse attention for extreme long-context
2. **Auxiliary-loss-free load balancing**: Theoretical analysis of DeepSeek's bias-term approach
3. **MTP computational efficiency**: Reduce overhead while maintaining 80-90% acceptance rates
4. **Quantization variance reduction**: Further minimize FP8 training accuracy gaps
5. **Safety and fairness**: Robust mechanisms for multilingual, demographically diverse deployments

---

## Conclusion

DeepSeek-V3 demonstrates that **hardware-software co-design can democratize frontier AI development**. By meticulously aligning architectural innovations (MLA, MoE), training optimizations (FP8, node-limited routing), and infrastructure decisions (MPFT topology, IBGDA) with H800 hardware constraints, the team achieved state-of-the-art performance at a fraction of typical costs.

The paper's most profound contribution is its **actionable roadmap for future hardware**:
- Configurable precision Tensor Cores with native fine-grained quantization
- Unified network adapters bridging scale-up/scale-out domains
- Memory-semantic communication with hardware-enforced ordering
- In-network computation for multicast/reduce operations
- DRAM-stacked accelerators for memory-bound MoE inference

As AI workloads continue scaling in complexity and size, the lessons from DeepSeek-V3 underscore that **efficiency through co-design, not brute-force scaling, will determine which organizations can afford to compete in the frontier AI race**. The open-source release of DeepEP, DeepGEMM, and DualPipe frameworks ensures these innovations benefit the entire community.

For academic researchers, systems engineers, and hardware architects, this paper is essential reading—it transforms abstract scaling laws into concrete, validated engineering principles that bridge the gap between algorithmic innovation and deployable systems.

---

**Paper Citation:** Zhao, C., Deng, C., Ruan, C., et al. (2025). Insights into DeepSeek-V3: Scaling Challenges and Reflections on Hardware for AI Architectures. *Proceedings of ISCA '25*, Tokyo, Japan. ACM.

**Open-Source Contributions:**
- DeepEP: [github.com/deepseek-ai/DeepEP](https://github.com/deepseek-ai/DeepEP)
- DeepGEMM: [github.com/deepseek-ai/DeepGEMM](https://github.com/deepseek-ai/DeepGEMM)
- DualPipe: [github.com/deepseek-ai/dualpipe](https://github.com/deepseek-ai/dualpipe)
- Fire-Flyer File System: [github.com/deepseek-ai/3FS](https://github.com/deepseek-ai/3FS)

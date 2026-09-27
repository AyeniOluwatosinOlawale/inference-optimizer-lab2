from __future__ import annotations


class WorkloadGenerator:
    """
    Generates prompts of approximate token length.
    Uses char-count approximation: 1 token ~= 4 chars.
    No tokenizer dependency.
    """

    TOPICS = [
        "distributed systems consensus algorithms",
        "neural network gradient descent optimization",
        "kubernetes pod scheduling strategies",
        "transformer attention mechanism internals",
        "database B-tree index structures",
        "GPU memory coalescing techniques",
        "compiler loop unrolling optimization",
        "cache coherence protocols in multicore CPUs",
        "bloom filters in key-value stores",
        "tensor parallelism for large language models",
        "RDMA networking for HPC clusters",
        "flash attention computation efficiency",
        "lock-free concurrent data structures",
        "prefix-sum scan GPU parallelism",
        "vector database approximate nearest neighbor",
        "key-value cache eviction policies",
        "speculative decoding for LLM acceleration",
        "model quantization INT8 vs FP8 tradeoffs",
        "continuous batching in vLLM",
        "PagedAttention KV cache management",
        "CUDA stream concurrency execution",
        "network topology fat-tree data centers",
        "microservice service mesh architecture",
        "JIT compilation vs AOT compilation",
        "memory-mapped files for large datasets",
        "graph neural network message passing",
        "reinforcement learning from human feedback",
        "diffusion model score matching",
        "mixture of experts sparse gating",
        "multi-head latent attention compression",
        "SIMD vectorized operations AVX-512",
        "distributed hash tables consistent hashing",
        "pipeline parallelism GPipe micro-batches",
        "zero redundancy optimizer ZeRO stages",
        "activation checkpointing memory savings",
        "beam search decoding strategies",
        "nucleus sampling temperature effects",
        "tokenizer byte-pair encoding BPE",
        "embedding model contrastive learning",
        "cross-attention in encoder-decoder architectures",
        "rotary positional embeddings RoPE",
        "grouped query attention GQA benefits",
        "KV cache quantization FP8 accuracy",
        "chunked prefill latency improvements",
        "HRRN scheduling for inference fairness",
        "radix tree prefix caching hit rates",
        "model sharding pipeline stages",
        "batch normalization vs layer normalization",
        "residual connections in deep networks",
        "mixture of depths adaptive compute",
    ]

    FILLER = (
        "You are an expert software engineer with deep knowledge of distributed systems, "
        "machine learning infrastructure, and high-performance computing. "
        "Always provide technically precise answers with concrete examples and trade-off analysis. "
    )

    def tokens_to_chars(self, tokens: int) -> int:
        return tokens * 4

    def make_prompt(
        self,
        context_tokens: int,
        workload: str,
        shared_prefix_tokens: int = 0,
        request_index: int = 0,
    ) -> str:
        """
        random: build a prompt of ~context_tokens using FILLER + topic sentences.
        shared_prefix: FILLER repeated to shared_prefix_tokens + unique tail question.
        """
        target_chars = self.tokens_to_chars(context_tokens)

        if workload == "shared_prefix":
            prefix_chars = self.tokens_to_chars(shared_prefix_tokens)
            # Build shared prefix by repeating FILLER
            prefix = ""
            while len(prefix) < prefix_chars:
                prefix += self.FILLER
            prefix = prefix[:prefix_chars]

            # Unique tail per request
            topic = self.TOPICS[request_index % len(self.TOPICS)]
            tail = (
                f"\n\nRequest #{request_index}: Please provide a comprehensive technical "
                f"explanation of {topic}, including implementation details, performance "
                f"characteristics, and real-world use cases. Compare alternative approaches "
                f"and discuss when each is most appropriate."
            )
            return prefix + tail

        else:
            # random workload: build from FILLER + topic sentences
            parts: list[str] = [self.FILLER]
            topic_idx = request_index % len(self.TOPICS)

            # Add varied topic content to reach target length
            while sum(len(p) for p in parts) < target_chars:
                idx = (topic_idx + len(parts)) % len(self.TOPICS)
                topic = self.TOPICS[idx]
                sentence = (
                    f"Explain the technical details and implementation considerations of {topic}, "
                    f"covering algorithmic complexity, memory access patterns, and hardware utilization. "
                )
                parts.append(sentence)

            text = "".join(parts)
            # Trim or extend to target_chars
            if len(text) > target_chars:
                text = text[:target_chars]

            # Append the actual question
            question_topic = self.TOPICS[topic_idx]
            text += (
                f"\n\nQuestion: Given the context above, provide a detailed technical analysis of "
                f"{question_topic} with specific focus on performance optimization, trade-offs, "
                f"and production deployment considerations."
            )
            return text

    def make_batch(
        self,
        n: int,
        context_tokens: int,
        workload: str,
        shared_prefix_tokens: int = 0,
    ) -> list[str]:
        return [
            self.make_prompt(context_tokens, workload, shared_prefix_tokens, i)
            for i in range(n)
        ]

# Living Memory for AI Agents
### The Dreamcatcher Architecture for Nightly Parametric Consolidation
#### Sleep-Inspired Re-Fine-Tuning with Organic Reinforcement

**Alex Skatell**
Topline, Inc. — Charleston, South Carolina
April 2026 | White Paper — Preprint

---

## Abstract

We introduce living memory for AI agent systems: memory that grows, consolidates, and evolves with the user rather than being statically stored and retrieved. Unlike retrieval-based approaches that look up information about a user from external databases, living memory is parametrically internalized—encoded in model weights through nightly re-fine-tuning, producing agents that genuinely know a user rather than agents that reference a dossier about them. We implement this concept through the Dreamcatcher architecture, which achieves living memory by constraining the memory model to sub-3B parameters and fully re-fine-tuning from fixed pretrained base weights every night on the complete, monotonically growing canonical memory ledger. This eliminates the sequential-update path dependence that drives catastrophic forgetting in incremental adaptation: each night’s model Mₜ = F(M₀, Tₜ) is an independent function of the fixed base weights and tonight’s rendered training set. We identify organic reinforcement as an emergent self-prioritization mechanism: important memories naturally accumulate more training pairs through real-world reuse, replicating frequency-dependent memory strengthening without any engineered priority system. The architecture produces a compact, structured context for injection into a frontier reasoning model’s prompt, with on-device inference ensuring that personal knowledge never leaves the user’s device. A dual-surface design provides a human-browsable knowledge vault for inspection and curation alongside the parametric layer for agent queries. We describe production reliability mechanisms including structured extraction schemas, automated recall benchmarking, advisory-only memory linting, checkpoint fallback, and disaster-recovery context injection. The architecture is model-agnostic, with Google’s Gemma 4 E2B (~2.3B effective parameters) as the default, at a marginal cost of approximately $0.05–$0.15 per nightly cycle.

Keywords: living memory, parametric memory, catastrophic forgetting, compact language models, AI agents, organic reinforcement, sleep consolidation, on-device inference, knowledge surface

## 1. Introduction

The problem of persistent memory in AI agent systems represents one of the most significant practical gaps between current LLM capabilities and the requirements of sustained human-AI collaboration. While frontier models demonstrate extraordinary reasoning within a single session, they fundamentally lack the capacity to accumulate knowledge about a specific user across sessions. Each new interaction begins from a blank state.

Existing approaches address this through non-parametric means: retrieval-augmented generation (RAG) with vector stores, tiered memory management (Letta/MemGPT[1], Mem0[5]), or structured context injection from files. A fourth approach, continual fine-tuning with LoRA[2], faces the well-documented challenge of catastrophic forgetting: sequential fine-tuning progressively degrades previously learned knowledge.[3]

This paper introduces the concept of living memory for AI agents—memory that is parametrically internalized into a compact model’s weights, growing and reconsolidating with each nightly training cycle rather than being statically stored and retrieved. We implement living memory through the Dreamcatcher architecture, which sidesteps catastrophic forgetting entirely by choosing a model small enough to fully re-fine-tune from fixed pretrained weights on every update cycle. We further show that two natural mechanisms—organic reinforcement through real-world reuse and deliberate semantic compression of older memories—manage dataset growth in a manner that directly parallels biological memory consolidation during sleep.

## 2. Related Work

### 2.1 Retrieval-Augmented Generation

RAG systems store memories as embeddings in vector databases and retrieve relevant passages at query time. This provides precise factual recall and selective deletion but introduces 50–200ms latency, suffers from embedding drift[4], and cannot internalize behavioral patterns. A RAG system can present examples of a user’s style; it cannot learn that style.

### 2.2 Tiered Memory Systems

Letta (MemGPT)[1] manages tiered memory through LLM tool calls. Mem0[5] provides a hybrid graph-vector-KV layer. Zep’s Graphiti[6] uses temporal knowledge graphs. These offer auditability but consume context tokens for every memory operation and cannot internalize knowledge into weights.

### 2.3 Continual Fine-Tuning with LoRA

LoRA[2] and QLoRA[7] enable parameter-efficient fine-tuning. However, the TRACE benchmark showed sequential LoRA fine-tuning across eight tasks reduced Llama-2-Chat’s GSM8K accuracy from 43% to 2%.[3] This has spawned work on orthogonal LoRA methods (SLAO[9]), experience replay (MSSR[10], SuRe[11]), adapter merging (TIES[12]), and selective updating (SPRInG[13]). Each reduces forgetting but introduces complexity and residual degradation.

### 2.4 Memory-Augmented Architectures

MemoryLLM[14] embeds self-updatable memory pools within transformer layers. IBM’s Larimar[15] adds brain-inspired episodic memory. MemoRAG[16] uses a small memory model for corpus-level understanding. Lamini’s memory tuning[17] trains millions of specialized LoRA experts. These require custom architectures or complex routing that cannot be applied to off-the-shelf models.

## 3. The Dreamcatcher Architecture

### 3.1 Biological Foundation

The architecture draws directly from the neuroscience of sleep-dependent memory consolidation. During waking hours, the hippocampus captures experiences rapidly in a fast-learning, episode-specific system. During slow-wave sleep, the hippocampus replays these experiences to the neocortex in compressed, interleaved form. The neocortex—a slow learner—gradually integrates the replayed experiences into its long-term knowledge web, strengthening connections and resolving conflicts with prior understanding. This process, termed memory consolidation, is one of the most well-established findings in cognitive neuroscience.

Dreamcatcher replicates each stage of this biological architecture for AI agents, as illustrated in Figure 1.

*Figure 1: The Dreamcatcher pipeline. Extraction (Step 1) calls the frontier LLM API on new sessions only. Training (Step 2) runs entirely locally on the full accumulated dataset. These are strictly separated operations.*

### 3.2 Formal Properties

We distinguish two objects in the nightly pipeline. Let Cₜ denote the canonical memory ledger at day t: the append-only set of extracted canonical facts (core_facts) in the database. Let Tₜ = R(Cₜ, πₜ) denote the rendered training set: the set of training pairs produced by applying rendering policy πₜ to the canonical ledger. Under the default policy, πₜ applies semantic compression (Section 5), reducing associative density for older facts while preserving every fact’s presence. Let M₀ denote the fixed pretrained base weights and F the re-fine-tuning procedure. The Dreamcatcher architecture produces each night’s model as:

Mₜ = F(M₀, Tₜ)    where    Tₜ = R(Cₜ, πₜ)    and    Cₜ ⊆ Cₜ₊₁

The model at day t is always an independent function of the fixed base weights and tonight’s rendered training set. There is no dependence on previous models: Mₜ is not derived from Mₜ₋₁. This eliminates the sequential-weight path dependence that drives catastrophic forgetting in incremental fine-tuning. We claim four structural properties and one system-level guarantee.

Monotonic corpus growth. The canonical ledger only grows: Cₜ ⊆ Cₜ₊₁. Every extracted canonical fact remains in the store permanently. No fact is deleted during normal operation. The rendering policy πₜ may adjust the associative density of older facts—reducing the number of training pair variations included in Tₜ—but all extracted pairs remain stored in SQLite. Semantic compression is a rendering policy that selects a subset for training, not a deletion operation. This preserves full auditability and rollback capability.

No replay forgetting. Under the append-only canonical ledger, every extracted memory is represented in every future training cycle from the fixed pretrained base. This eliminates the class of forgetting caused by sequential-update path dependence. Note that this is a guarantee about the training process, not a theorem about per-query recall accuracy—a finite model can still regress on specific facts due to capacity limits, contradictory data, or stochastic optimization, even when the relevant training examples remain in the rendered set.

Replayable rebuild. If the canonical ledger contains erroneous data, correcting or removing the offending records and re-fine-tuning from M₀ produces a model with zero residual contamination from the error. In incremental systems, erroneous data that influenced subsequent training sessions leaves an irrecoverable trace in the weights. This property supports right-to-erasure compliance: deleting memories from the canonical store, purging associated artifacts (raw transcripts, extracted JSON, cached model checkpoints), and retraining produces a clean model.

Full auditability. The SQLite database serves as a complete, human-readable record of every fact the model has been trained on. Every memory traces to a specific session, extraction, and training pair. This enables debugging, compliance, and reproducibility (given M₀, Cₜ, and πₜ, the training is reproducible up to hardware non-determinism).

System-level recoverability. During stale-model failures—when the nightly pipeline fails and the deployed model lacks recently extracted memories—the canonical store provides a backstop: the inference server retrieves untrained recent memories and injects them as structured context (Section 6.4). This ensures that recent memories remain recoverable even when the parametric model has not yet been trained on them.

### 3.3 Base Model Selection and Hardware Considerations

The Dreamcatcher architecture is model-agnostic by design: any sub-2B dense model with adequate instruction-following capability can serve as the base. The choice of base model is a one-line configuration change that does not affect the training pipeline, database schema, or inference server. We evaluate three candidates and recommend a tiered approach based on hardware availability and fine-tuning maturity.

Qwen3.5-0.8B (Alibaba, 2026) is the recommended default. As a dense 800M parameter model with no architectural complexity, its full fine-tuning characteristics are well-understood: training fits comfortably within 8–12GB VRAM and completes in 2–3 minutes on an RTX 3090 for typical dataset sizes. It offers strong instruction following, supports 262K context length, and has extensive community tooling support. For the majority of users, Qwen3.5-0.8B provides the best balance of capability, reliability, and hardware compatibility.

Gemma 4 E2B (Google, April 2026) offers stronger structured-output behavior for function-calling-style prompts and agentic workflow support, making it an aspirational option for users who want the highest-quality memory model.[19] However, an important nuance affects fine-tuning feasibility. The “E” in E2B stands for “effective”: the model’s computational core is approximately 2B parameters, but it incorporates Per-Layer Embeddings (PLE)—per-layer token lookup tables that inflate the total parameter count beyond 2B. For Dreamcatcher’s use case, the PLE tables can be frozen during fine-tuning (vocabulary does not change when encoding personal memories), reducing trainable parameters to the ~2B core. However, PLE is a novel architecture with no published fine-tuning benchmarks as of this writing. Users should verify VRAM feasibility on their target hardware before adopting Gemma 4 E2B.

SmolLM2-360M (HuggingFace) serves as the ultra-light option for CPU-only machines or extremely constrained hardware. At 360M parameters, full fine-tuning completes in under 90 seconds on a GPU but produces lower-quality structured output than the other candidates.

The key architectural claim—that compact models are small enough for nightly full re-fine-tuning while retaining sufficient linguistic competence for structured memory retrieval—holds across all three candidates. The model-agnostic design ensures that as new compact models are released, Dreamcatcher benefits automatically from improved base models without architectural changes.

*Table 1: Training Time Estimates for Full Re-Fine-Tuning (3 epochs)*

*Gemma 4 E2B training times are estimates pending empirical verification of PLE fine-tuning characteristics.

### 3.4 Extraction and Training: A Critical Separation

A common misconception about the Dreamcatcher architecture is that the frontier LLM API is called on the entire accumulated dataset every night. This is incorrect, and the distinction is critical for understanding both the cost model and the latency profile of the system.

The pipeline contains two strictly separated operations. The extraction step calls the frontier LLM API to process only new, unprocessed session transcripts from the current day. Each session is extracted exactly once, its structured memories and training pairs are stored permanently in a SQLite database, and it is flagged as processed. Historical sessions are never re-sent to the API. The training step runs entirely locally on the user’s GPU, loading all training pairs from the SQLite database (both today’s new pairs and all historical pairs) and fine-tuning the base model from scratch. No API calls are made during training. No network access is required.

This separation means that the frontier LLM API cost scales only with the volume of new daily sessions (typically $0.01–$0.05), regardless of whether the database contains 100 or 50,000 historical training pairs. The only cost that scales with historical data volume is local GPU training time—a compute resource with zero marginal monetary cost for users with local hardware.

### 3.5 On-Device Deployment and Privacy Model

The Dreamcatcher memory model is designed for on-device inference. The trained model (700MB–1.6GB depending on base model and quantization) runs directly on the user’s laptop, phone, or workstation—not on a remote server. When an agent queries the memory model, the query is a local computation: a forward pass through a small model already loaded in device memory, requiring no network access.

This creates a clean separation between personal knowledge (which stays on-device) and reasoning (which may go to the cloud). When a user asks their agent “What should I prioritize this week?”, the agent first queries the local Dreamcatcher model, which returns structured JSON—perhaps noting a compressed work deadline, a Thursday pediatrician appointment, and a preference for front-loading deep work. That structured context block is then appended to the system prompt sent to the cloud frontier model. The frontier model receives a curated summary of what’s relevant to this specific query. It never sees the raw session transcripts, the full memory database, or the training data. The user’s complete personal history stays on their device; only a handful of structured facts leave it per interaction.

*Figure 2: On-device deployment. The memory model runs locally; only curated structured context leaves the device. The frontier model in the cloud sees relevant facts, never raw personal data.*

Training and inference can occur on different machines. The nightly training pipeline requires a GPU and runs on the user’s workstation or a cloud instance. The resulting model file is distributed to all of the user’s devices—laptop, phone, tablet—via file sync (cloud storage, local network, or manual copy). Each device loads the same freshly trained model the next morning. This naturally supports multi-device, multi-agent use: sessions from OpenClaw on a laptop and Hermes Agent on a phone are collected centrally, trained into one model overnight, and the resulting memory model is distributed back to all devices.

This on-device architecture materially strengthens the privacy posture. Membership inference attacks against fine-tuned models[22] require the ability to query the model remotely. When the model never leaves the user’s physical device and never serves network requests, the attack surface reduces to physical device access—the same threat model that protects the user’s photos, messages, and banking applications.

One important caveat: the extraction step (Section 3.4) sends raw session transcripts to a frontier LLM API for memory extraction. During this step, personal data—including potentially sensitive medical, family, or business context—is transmitted to a cloud provider. The accurate privacy claim is therefore: inference is local, nightly model training is local, but extraction is cloud-based unless replaced with a local extraction model. For users requiring full data sovereignty, a smaller local model can perform extraction at the cost of lower extraction quality. The default configuration prioritizes extraction quality over full local processing.

## 4. Organic Reinforcement: Self-Prioritizing Memory

A central concern with any memory system is prioritization: how does the system ensure that important facts receive robust encoding while routine details do not consume disproportionate capacity? Existing approaches address this through engineered scoring systems, priority queues, or manual curation. The Dreamcatcher architecture reveals an emergent alternative that requires no engineering.

Consider the lifecycle of two different facts in a system that extracts training pairs from real agent sessions over months of use. A child’s food allergy is important. It appears not once but repeatedly: during a pediatrician consultation session, again when planning a birthday party, again when researching schools, again when booking restaurants for a family trip. Each session generates 3–5 fresh training pairs from naturally distinct contexts—“What food restrictions matter for the user’s family?” from the party session, “What medical information is relevant for the user’s children?” from the school session. These are not synthetic reformulations of the same question; they are organically different questions arising from genuinely different real-world situations that happen to reference the same underlying fact.

Conversely, the name of a contractor who performed a one-time repair six months ago appeared in exactly one session and generated its initial 3–5 training pairs. Those pairs remain in the dataset permanently. The model still trains on them every night. The fact is not forgotten. But it occupies a naturally small footprint relative to the allergy—perhaps 4 training pairs versus 28—because it was never discussed again.

*Figure 3: Organic reinforcement. Important facts accumulate training pairs through natural real-world reuse. The model allocates proportionally more capacity to frequently-referenced knowledge without any engineered priority mechanism.*

This mechanism directly parallels the neuroscience of frequency-dependent memory strengthening. Memories that are reactivated across many different contexts develop stronger cortical representations through repeated consolidation cycles, while one-off experiences persist at minimal encoding density. The Dreamcatcher architecture replicates this effect structurally: the dataset’s own composition serves as the priority signal, and the real world provides it without engineering. Dreamcatcher uses organic reinforcement as the default salience prior, with optional pinning or manual curation for rare but mission-critical facts—such as allergies, legal deadlines, medication reactions, or do-not-contact instructions—that may appear in only one session but must never receive degraded encoding.

## 5. Semantic Compression and Scaling

As the canonical memory corpus grows, so does the nightly training set. Semantic compression manages this growth by reducing the associative density of older memories—the number of training pair variations used to encode each fact—while preserving every canonical fact in the corpus. This replicates the neuroscientific episodic-to-semantic transition: a fresh memory is encoded through multiple contextual pathways (4–5 training pairs), while an older memory is consolidated into leaner semantic encoding (1–2 pairs retaining only the most general-purpose questions). Critically, no canonical fact is removed. The formal guarantee of monotonic corpus growth (Dₜ ⊆ Dₜ₊₁) holds because Dₜ is defined as the set of extracted facts, not training pairs.

The computational impact is significant. For every 10,000 historical facts with an original 4–5 training pairs each, the compressed training set contains approximately 15,000–20,000 pairs instead of 40,000–50,000. This reclaims roughly 60% of historical training overhead, approximately doubling the time horizon before dataset size threatens the nightly window.

Organic reinforcement (Section 4) interacts favorably with compression. A fact discussed across five sessions in the past six months has accumulated 20+ fresh training pairs from those recent sessions—all at full density because they are recent. Even if its original pairs from seven months ago are compressed to 1–2, the total training pair count for that fact increases over time rather than decreasing. Compression primarily affects genuinely one-off memories that were never discussed again—precisely those that should receive leaner encoding.

Temporal supersession is supported linguistically rather than through explicit metadata. The extraction prompt (Section 6.1) injects the session’s exact date into the frontier model’s instructions, encouraging every extracted fact to carry temporal markers in the training tokens (e.g., “As of April 3, 2026, the user routes through Qwen”). When the model trains on both an older and a newer version of a fact, the natural language of recency provides a helpful cue for the model to learn which represents the current state. This mechanism works well for simple factual updates but is not guaranteed to resolve complex evolving beliefs or multi-timescale contradictions; for mission-critical temporal distinctions, users should verify current facts through the curation workflow or pinning mechanism.

*Table 2: Projected Nightly Training Times (Qwen3.5-0.8B, RTX 3090, 3 epochs, with compression)*

Even at the five-year mark, nightly training completes well within an overnight window. Hardware improvements (30–50% per GPU generation, approximately every 18–24 months) further extend this ceiling. For users who require the strictest formal guarantees, the codebase supports an uncompressed mode where all training pairs are retained at full density; the only cost is proportionally longer training times (approximately 2–2.5× the compressed estimates above).

## 6. Production Reliability Mechanisms

A practical memory system must handle failure modes gracefully. We describe four mechanisms—structured extraction, automated recall benchmarking, checkpoint fallback, and disaster-recovery context injection—that ensure Dreamcatcher operates reliably in production. These mechanisms address the primary concern that a fully re-fine-tuned model is only as good as tonight’s training run, and that training can fail silently.

### 6.1 Structured Extraction Schema

The multi-angle training approach depends on the extraction prompt generating training pairs in a consistent generality hierarchy: broad questions first, specific questions last. Rather than relying on prompt compliance alone, we enforce the hierarchy structurally through the JSON extraction schema.

The extraction prompt requires the frontier model to output each memory as a structured object containing a core_fact field—a single canonical statement of the essential truth—followed by explicitly keyed training pairs descending from semantic (the broadest formulation) through contextual to specific. The JSON key names physically force the generality ordering. The collector strips these into indexed pairs during storage. This converts a soft instruction into a hard structural constraint that the model cannot violate without producing invalid JSON.

The core_fact field serves a second critical purpose: it provides a ground-truth reference for the automated recall benchmark described below.

*Figure 4: The structured extraction schema forces generality ordering through JSON key names. The core_fact provides a ground-truth target for post-training recall verification.*

### 6.2 Automated Recall Benchmark

After each nightly training run, an automated benchmark verifies that the newly trained model has successfully internalized the training data. The benchmark randomly selects 20–30 training examples where pair_index = 0 (the most general formulation of each fact), queries the freshly trained model with these questions, and computes embedding similarity between the model’s responses and the corresponding core_fact values stored during extraction.

The recall score—the mean cosine similarity across the benchmark set—is logged with each training run. This produces a longitudinal time series that monitors coarse retention and training health over time: it detects training failures before the bad model is deployed, enables early detection of capacity saturation as the dataset grows, and provides a baseline measure of recall stability across nights. Because the benchmark runs entirely locally against the freshly trained model, it adds approximately 30–60 seconds to the nightly pipeline and incurs zero API cost. A rigorous research evaluation would require stronger methodology: held-out paraphrase prompts not seen during training, exact JSON field-level accuracy scoring, knowledge-update and contradiction cases, rare-but-critical one-shot memories, false-premise abstention tests, and end-to-end agent-task comparisons against RAG and continual LoRA baselines. These are planned as future work once longitudinal deployment data is available.

If the recall score falls below a configurable threshold (default: 0.75 mean cosine similarity), the pipeline does not execute the symlink swap that would deploy the new model. The current pointer continues to reference the previous night’s model, ensuring that agents never interact with a model that failed its quality check.

### 6.3 Weekly Checkpoint Fallback

The symlink-swap mechanism provides single-night rollback: if tonight’s training fails the benchmark, last night’s model continues serving. For multi-night failures—such as a corrupted dataset, a broken training dependency, or a base-model download issue—a deeper fallback is required. Each Sunday night, the pipeline copies the successfully benchmarked model to a golden/ directory, maintaining a rolling month of weekly checkpoints at approximately 700MB–1.6GB per checkpoint depending on the base model. If the nightly benchmark fails for multiple consecutive nights, the system falls back to the most recent golden checkpoint.

### 6.4 Disaster Recovery: Delta Context Injection

When a checkpoint fallback is triggered—for example, Thursday’s agents are served by Sunday’s golden model because Monday through Wednesday’s training runs all failed their benchmarks—the model has a three-day parametric knowledge gap. Memories extracted from Monday, Tuesday, and Wednesday exist in the SQLite database but have not been trained into any deployed model.

The disaster recovery mechanism bridges this gap temporarily. The inference server detects the timestamp delta between the active model’s training date and the current date. If the delta exceeds 24 hours, it queries the database for all core_fact entries created after the model’s training date and injects them as a structured JSON block into the agent’s system prompt alongside the model’s parametric output. For a typical three-day gap, this adds approximately 10–20 core facts—a negligible fraction of the context window.

This mechanism is strictly quarantined as disaster recovery. It activates only when the model serving timestamp is stale. During normal operation—when the nightly pipeline succeeds and the symlink swap deploys a fresh model—no context injection occurs. The architectural discipline is preserved: agents experience during the day, the model consolidates at night, and the delta injection exists solely as a safety net against multi-day training failures. This design accepts a temporary, bounded compromise (retrieval-style context injection for a few days of facts) to maintain the user’s uninterrupted experience while the training pipeline is repaired.

### 6.5 Browsable Knowledge Surface

A memory system that cannot be inspected by its user is a memory system that cannot be trusted. The canonical ledger in SQLite is technically queryable but practically opaque—a user cannot browse their memories, spot extraction errors, or understand how their knowledge is organized without writing SQL queries. To address this, the nightly pipeline exports the canonical ledger as a structured directory of markdown files compatible with knowledge management tools such as Obsidian, Logseq, or any markdown editor.

Each category of memory (projects, preferences, facts, patterns, relationships, decisions) becomes a markdown file containing the core_facts for that category, the session dates they were extracted from, and cross-references to related facts in other categories. Each entry carries a stable memory_id and structured metadata in YAML frontmatter, enabling controlled round-trip synchronization. The result is a human-readable, browsable projection of everything the system knows about the user. Critically, the SQLite canonical ledger remains the sole source of truth; the markdown vault is a curated view, not a competing data store. User corrections operate through structured actions (marking entries as deprecated, flagging for deletion, adding notes) that map to explicit canonical-store operations, avoiding the bidirectional drift that would result from unconstrained freeform editing. This transforms the user curation problem from “query the database” to “browse the vault,” and provides a natural interface for right-to-erasure compliance: mark an entry for deletion, retrain, and the model has zero trace of the deleted information.

This dual-surface approach—a human-readable knowledge vault by day, a compact parametric memory layer by night—was independently motivated by Karpathy (2026), who described using LLMs to compile raw source documents into structured markdown knowledge bases that the LLM maintains and the human browses.[34] Karpathy independently highlights the value of LLM-maintained knowledge compilation and explicitly points toward parametric internalization as a natural next step, noting that “the natural desire is to also think about synthetic data generation + finetuning to have your LLM know the data in its weights instead of just context windows.” The Dreamcatcher architecture implements this next step: the vault serves as the human-readable knowledge surface for inspection and curation, while the nightly re-fine-tuned model serves as the agent-queryable parametric layer for instant structured recall.

### 6.6 Memory Linting

The extraction step (Section 3.4) is a single-point-of-failure for data quality: if the frontier LLM misinterprets a transcript or hallucinates a fact during extraction, the error is permanently baked into the canonical ledger and trained into every subsequent model. Organic reinforcement may amplify the error if the flawed fact gets referenced in future sessions. The automated recall benchmark (Section 6.2) catches model-level training failures but cannot detect errors that are factually plausible yet wrong.

To address this, a periodic memory lint pass—running weekly rather than nightly to control API cost—reviews batches of canonical facts for internal consistency. A rule-based pre-pass handles deterministic checks (exact duplicates, date conflicts, conflicting scalar values) without API cost. The frontier LLM then handles the fuzzy layer: identifying semantic contradictions, likely superseded information, implausible extractions, missing companion facts, and candidate abstractions. The output is an advisory lint report—saved as markdown in the knowledge surface (Section 6.5)—that flags issues for human review. Critically, the linter never silently mutates memory; it produces typed findings (contradiction, likely stale, likely duplicate, missing companion, weakly supported inference) with confidence scores and source citations to the underlying facts and sessions. The user reviews, approves, or dismisses each finding, and only confirmed corrections are applied to the canonical ledger for propagation to the next nightly retrain.

## 7. Comparative Analysis

*Table 3: Architectural Comparison Across Memory Approaches*

Dreamcatcher is designed to combine no sequential-update drift, high behavioral personalization, full rollback and auditability, and self-prioritizing memory density in a single architecture. The addition of organic reinforcement as a row in this comparison highlights a property that no existing system achieves: automatic, engineering-free priority allocation through the natural dynamics of user interaction. Whether factual precision matches or exceeds retrieval-based approaches remains an empirical question to be validated through deployment (Section 6.2).

## 8. Novelty and Contributions

We acknowledge that full retraining as a strategy for avoiding catastrophic forgetting is the established baseline in the continual learning literature—Van de Ven et al. (2024) note that practitioners frequently retrain when computational budgets permit.[20] The entire continual learning field exists because this approach is computationally prohibitive for large models. Our contribution is not the mechanism of full retraining itself but the identification that full retraining becomes practical and cost-effective at compact sub-2B parameter scale for the personal memory use case—a design point the field has overlooked. We identify five specific contributions.

First, the architectural argument that constraining model size to enable full retraining is a superior design strategy for personal AI memory compared to incremental continual learning at larger scale. While Personal.ai has deployed per-user language models commercially,[23] their architecture is proprietary and does not describe the full-retrain-from-scratch mechanism. No published system proposes nightly full retraining of a sub-2B parameter model on accumulated personal data as an open architecture for AI agents.

Second, organic reinforcement as an emergent self-prioritization mechanism. While frequency-dependent memory strengthening is a known property of neural network training, and MemoryBank[24] implements explicit frequency-based scoring for retrieval memory, the observation that the Dreamcatcher architecture achieves equivalent prioritization as a zero-cost emergent property of its data pipeline—requiring no scoring system—has not been previously described.

Third, multi-angle synthetic training pair generation for robust factual encoding, supported by Yang et al.’s finding that diverse representations are critical for fact learning in LLMs.[25]

Fourth, the honest framing of the architecture as a parametric memory layer that produces compact structured context for prompt-time injection into a frontier reasoning model, with on-device inference ensuring personal knowledge stays local. This hybrid delivery mechanism—parametric internalization for knowledge storage, prompt injection for knowledge delivery—is distinct from both pure RAG (no internalization) and pure fine-tuning of the reasoning model itself (impractical for personal data).

Fifth, the production reliability suite (Section 6)—structured extraction, automated recall benchmarking, checkpoint fallback, and disaster-recovery context injection—which provides a complete operational framework for deploying parametric memory systems. The automated recall benchmark in particular produces the longitudinal empirical data needed to validate the architecture’s claims over time.

We note several adjacent lines of work that share partial overlap. The ENGRAM system provides typed lightweight memory orchestration for AI agents but uses external storage rather than parametric encoding. The PRIME framework implements cognitive dual-memory personalization but does not employ full re-fine-tuning from fixed weights. RAG-Tuned-LLM approaches use fine-tuning to create “LLM-native memory” but rely on incremental updates subject to sequential drift. Our novelty is not parametric memory in general, but the specific combination of a compact dedicated memory model, nightly full re-fine-tuning from fixed pretrained initialization, an append-only canonical memory ledger, and a production reliability suite for deployment.

## 9. Limitations and Future Work

Hallucination risk. Gekhman et al. (EMNLP 2024) demonstrated that fine-tuning LLMs on new knowledge—information absent from the base model’s pretraining data—linearly increases the model’s tendency to hallucinate.[21] All personal memories constitute “new knowledge” by definition. This is the most significant theoretical concern for the Dreamcatcher architecture. We hypothesize that three design features mitigate this risk: the structured JSON output constraint limits the model’s generation freedom (reducing opportunity for hallucination), the multi-angle training provides multiple reinforcement pathways per fact (strengthening accurate recall), and the automated recall benchmark (Section 6.2) detects hallucination-induced accuracy degradation before deployment. However, rigorous empirical measurement of hallucination rates in Dreamcatcher-trained models remains critical future work.

Privacy vulnerability. Fine-tuned language models are susceptible to membership inference attacks. Fu et al. (NeurIPS 2024) achieved AUC of 0.9 on membership inference against fine-tuned LLMs,[22] with vulnerability positively correlated with trainable parameter count. A Dreamcatcher model encodes an individual’s complete personal history in its weights, creating a concentrated privacy target. However, the on-device deployment model (Section 3.5) substantially mitigates this risk: membership inference attacks require the ability to query the model, and a model that never leaves the user’s physical device and never serves network requests reduces the attack surface to physical device access—the same threat model that protects the user’s photos and banking applications. For any deployment where the model is exposed to network queries, differential privacy during training or selective data obfuscation should be considered.

Scaling ceiling. With semantic compression active on the recommended Qwen3.5-0.8B model, nightly training time reaches approximately 10–15 minutes at the five-year mark (Table 2). This fits comfortably within a nightly window and will compress further as GPU throughput improves. The architecture’s scaling profile is robust for the foreseeable future of single-user personal memory workloads.

Absence of empirical evaluation. This paper presents an architectural proposal with theoretical analysis but no experimental results. The automated recall benchmark (Section 6.2) is designed to produce longitudinal empirical data during deployment, and we intend to publish recall accuracy curves, hallucination measurements, and comparisons against RAG and continual LoRA baselines in a follow-up study. A head-to-head comparison against RAG on personal memory tasks is particularly important, as RAG offers immediate updates, exact-text retrieval, and trivial deletion for privacy compliance—advantages that parametric memory cannot match on specific factual queries.

Extraction quality bottleneck. The dependence on a frontier LLM for memory extraction means that if the extraction model misidentifies or omits a relevant memory, it is permanently absent from training data. The browsable knowledge surface (Section 6.5) and memory linting (Section 6.6) partially address this by making errors visible and flagging inconsistencies, but user-facing curation tools remain an important development priority. The mixture-of-experts extension—training separate small models per memory category—represents a promising direction for improving factual precision within the full-retrain paradigm.

Convergent knowledge surface approaches. Independent of this work, Karpathy (2026) described using LLMs to compile raw source documents into structured markdown knowledge bases—“wikis” that the LLM maintains and the human browses—and noted that the natural evolution is “synthetic data generation + finetuning to have your LLM know the data in its weights instead of just context windows.”[34] This convergence suggests that the combination of a human-browsable knowledge surface with a parametric memory model represents a natural architecture that multiple practitioners are arriving at from different directions. Future work should explore tighter integration between these layers, including the possibility that agent queries which synthesize multiple memories could propose candidate abstractions for the canonical ledger. However, such query-generated enrichment requires strict safeguards against self-reinforcing hallucination: if the model synthesizes from imperfect memory and the synthesis is promoted back into training data, future syntheses build on potentially flawed foundations and the system drifts toward confident narrative rather than grounded truth. A memory class system—distinguishing observed facts (extracted directly from sessions), inferred patterns (detected across sessions), and synthesized summaries (produced by queries)—with explicit promotion policies requiring user confirmation or recurrence before training, would mitigate this risk while preserving the value of query-driven knowledge enrichment.

## 10. Conclusion

We have presented living memory for AI agent systems—memory that grows, consolidates, and evolves with the user through nightly parametric internalization—and the Dreamcatcher architecture that implements it. By fully re-fine-tuning a compact model from fixed pretrained weights on the complete, monotonically growing memory dataset each night, the architecture provides structural properties—no sequential path dependence, perfect rollback, full auditability, and monotonic dataset accumulation—that incremental fine-tuning approaches cannot offer. Organic reinforcement provides automatic, engineering-free priority allocation through the natural dynamics of real-world interaction, where important facts accumulate denser training pair representations simply by being discussed more often.

The architecture is deliberately hybrid: a compact on-device model internalizes behavioral patterns and personal knowledge into its weights, then produces structured context for prompt-time injection into a frontier reasoning model. This separation—parametric internalization for storage, prompt injection for delivery—gives agents the feeling of genuinely knowing a user while preserving the frontier model’s full reasoning capability. A dual-surface design provides a human-browsable knowledge vault for inspection and curation alongside the parametric layer for agent queries. Personal knowledge stays on-device during inference; only curated structured summaries reach the cloud.

The biological brain solved the problem of integrating new experiences without overwriting old knowledge millions of years ago. It did not solve it with clever incremental update rules. It solved it by replaying everything and rebuilding consolidated representations each sleep cycle. Dreamcatcher follows the same logic—and the same name. It catches the day’s experiences and weaves them into lasting, living memory during the night.

## References

[1] Packer, C., et al. (2023). MemGPT: Towards LLMs as Operating Systems. arXiv:2310.08560.

[2] Hu, E. J., et al. (2022). LoRA: Low-Rank Adaptation of Large Language Models. ICLR 2022.

[3] Wang, X., et al. (2023). TRACE: A Comprehensive Benchmark for Continual Learning in LLMs. arXiv:2310.06762.

[4] Decompressed.io (2025). Detecting Embedding Drift: The Silent Killer of RAG Accuracy.

[5] Mem0 (2024). Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory. arXiv:2504.19413.

[6] Zep AI (2024). Graphiti: Build Real-Time Knowledge Graphs for AI Agents. github.com/getzep/graphiti.

[7] Dettmers, T., et al. (2023). QLoRA: Efficient Finetuning of Quantized Language Models. NeurIPS 2023.

[8] Biderman, D., et al. (2024). LoRA Learns Less and Forgets Less. TMLR.

[9] Qiao, Y. & Mahdavi, M. (2024). SLAO: Single LoRA Continual Learning via Continual Merging. arXiv:2512.23017.

[10] MSSR (2025). Memory-Aware Adaptive Replay for Continual LLM Fine-Tuning. arXiv:2603.09892.

[11] SuRe (2025). Surprise-Driven Prioritised Replay for Continual LLM Learning. arXiv:2511.22367.

[12] Yadav, P., et al. (2023). TIES-Merging: Resolving Interference When Merging Models. NeurIPS 2023.

[13] SPRInG (2025). Continual LLM Personalization via Selective Parametric Adaptation. arXiv:2601.09974.

[14] Wang, Y., et al. (2024). MemoryLLM: Towards Self-Updatable Large Language Models. ICML 2024.

[15] IBM & Princeton (2024). Larimar: Large Language Models with Episodic Memory Control. ICML 2024.

[16] Qian, H., et al. (2025). MemoRAG: Next-Gen RAG Via Memory-Inspired Knowledge Discovery. TheWebConf 2025.

[17] Lamini AI (2024). Introducing Lamini Memory Tuning: 95% LLM Accuracy, 10x Fewer Hallucinations.

[18] Ovadia, O., et al. (2024). Fine-Tuning or Retrieval? Comparing Knowledge Injection in LLMs. EMNLP 2024.

[19] Google DeepMind (2026). Gemma 4: Byte for Byte, the Most Capable Open Models. April 2, 2026.

[20] Van de Ven, G. M., Tuytelaars, T., & Tolias, A. S. (2024). Continual Learning and Catastrophic Forgetting. arXiv:2403.05175.

[21] Gekhman, Z., et al. (2024). Does Fine-Tuning LLMs on New Knowledge Encourage Hallucinations? EMNLP 2024.

[22] Fu, Z., et al. (2024). Membership Inference Attacks against Fine-tuned Large Language Models via Self-prompt Calibration. NeurIPS 2024.

[23] Personal.ai (2024). Personal Language Models: Per-User Fine-Tuning for AI Memory. personal.ai.

[24] Zhong, W., et al. (2024). MemoryBank: Enhancing Large Language Models with Long-Term Memory. AAAI 2024.

[25] Yang, Z., et al. (2025). Synthetic Continued Pretraining. ICLR 2025.

[26] McClelland, J. L., McNaughton, B. L., & O'Reilly, R. C. (1995). Why there are complementary learning systems in the hippocampus and neocortex. Psychological Review, 102(3), 419–457.

[27] Kumaran, D., Hassabis, D., & McClelland, J. L. (2016). What learning systems do intelligent agents need? Complementary learning systems theory updated. Trends in Cognitive Sciences, 20(7), 512–534.

[28] Diekelmann, S. & Born, J. (2010). The memory function of sleep. Nature Reviews Neuroscience, 11(2), 114–126.

[29] Klinzing, J. G., Niethard, N., & Born, J. (2019). Mechanisms of systems memory consolidation during sleep. Nature Neuroscience, 22(10), 1598–1610.

[30] Wilson, M. A. & McNaughton, B. L. (1994). Reactivation of hippocampal ensemble memories during sleep. Science, 265(5172), 676–679.

[31] Sekeres, M. J., et al. (2023). Time-dependent memory transformation is semantic in nature. Nature Communications, 14, 6004.

[32] Wang, L., et al. (2023). Augmenting Language Models with Long-Term Memory. NeurIPS 2023.

[33] Kandpal, N., et al. (2023). Large Language Models Struggle to Learn Long-Tail Knowledge. ICML 2023.

[34] Karpathy, A. (2026). LLM Knowledge Bases. x.com/karpathy, April 3, 2026.

# ml-platform (fictional fixture)

An in-house ML platform: training on GKE A100 nodes, experiment tracking in MLflow, inference served by TGI, vector search on Qdrant.

> **Not a real project.** This fixture exists so ID-RAG parallel benchmarks can test decision recall on a plausible ML-infrastructure stack. All decisions in `.continuity/decisions.json` are fictional.

## Stack at a glance

- **Framework**: PyTorch 2.4
- **Orchestration**: Argo Workflows on GKE
- **Experiment tracking**: MLflow (self-hosted)
- **Feature store**: Feast
- **Serving**: Hugging Face TGI (replaced vLLM)
- **Vector DB**: Qdrant (replaced pgvector)
- **Fine-tuning**: LoRA adapters (PEFT)
- **Drift monitoring**: Evidently

## Supersede chains

- `mlp-serving-vllm` → superseded by `mlp-serving-tgi`
- `mlp-vector-pgvector` → superseded by `mlp-vector-qdrant`

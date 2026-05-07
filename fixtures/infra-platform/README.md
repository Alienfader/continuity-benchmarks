# infra-platform (fictional fixture)

The internal Kubernetes platform that hosts every other service at the company: EKS clusters, GitOps via ArgoCD, Linkerd service mesh, and a unified observability stack (Prometheus + Tempo + Grafana).

> **Not a real project.** This fixture exists so ID-RAG parallel benchmarks can test decision recall on a plausible platform-engineering stack. All decisions in `.continuity/decisions.json` are fictional.

## Stack at a glance

- **Clusters**: EKS
- **Service mesh**: Linkerd
- **Ingress**: Traefik
- **Secrets**: HashiCorp Vault
- **GitOps**: ArgoCD (apps + infra)
- **Registry**: ECR (containers), Artifactory (language packages)
- **Metrics**: Prometheus + Thanos
- **Logs**: Vector → Elastic (replaced Loki)
- **Tracing**: Tempo (replaced Jaeger)
- **CI**: Buildkite
- **Policy**: OPA Gatekeeper
- **Autoscaling**: KEDA

## Supersede chains

- `infra-logs-loki` → superseded by `infra-logs-vector-elastic`
- `infra-tracing-jaeger` → superseded by `infra-tracing-tempo`

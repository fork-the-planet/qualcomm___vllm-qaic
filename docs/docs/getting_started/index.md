# Getting Started

!!! success "New here?"
    Start with the [Quick Start](quickstart.md) — zero to first inference in under 5 minutes using Docker.

Choose your path based on your use case:

| Goal | Mode | Start here |
|------|------|-----------|
| **Production serving** with maximum throughput | AOT | [AOT Installation](installation/aot_mode.md) |
| **Development / prototyping** with flexibility | PYT (Eager) | [PYT Installation](installation/pyt_mode.md) |
| **Fastest path to first inference** | AOT Docker | [Quickstart](quickstart.md) |

## Which Mode Should I Use?

| Criterion | AOT (Ahead-of-Time) | PYT (Eager) (Experimental) |
|-----------|---------------------|-------------|
| Compilation | Models pre-compiled to QPCs via QEfficient | Dynamic execution via torch_qaic |
| Latency | Optimized static graphs | Runtime flexibility |
| Features | Full: SpD, LoRA, disaggregated, multimodal, embedding | Selected VLMs |
| Best for | Production workloads | R&D, new model bring-up |

!!! tip "Recommendation"
    If you are deploying models to serve production traffic, start with **AOT mode**.
    If you are experimenting with new model architectures or need rapid iteration, use **PYT mode**.

## Prerequisites

Before installing either mode, ensure you have:

- Qualcomm Cloud AI hardware (Cloud AI 100, Cloud AI 080, or Cloud AI 100 Ultra)
- Linux (Ubuntu 22.04+)
- [Qualcomm Cloud AI SDK](https://quic.github.io/cloud-ai-sdk-pages/latest/Getting-Started/Installation/index.html) >= 1.22.0
- Python 3.12

See [Prerequisites](installation/prerequisites.md) for detailed setup.

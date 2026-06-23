# Embedding Models

Embedding networks transform high-dimensional inputs — text, images, items — into dense, low-dimensional vectors that capture semantic relationships. These vectors enable tasks such as similarity search, reranking, classification, and recommendation.

## Supported Models

| Model | Notes |
|---|---|
| `intfloat/multilingual-e5-large` | |
| `intfloat/e5-large` | |
| `jinaai/jina-embeddings-v2-base-en` | Requires `trust_remote_code=True`; see patch note below |
| `jinaai/jina-embeddings-v2-base-code` | Requires `trust_remote_code=True` |

!!! note "Limitation"
    `sentence-transformers/gtr-t5-large` is not supported.

## Usage

```python
from vllm import LLM

# CPU pooling — compile for multiple sequence lengths
model = LLM(
    model="intfloat/multilingual-e5-large",
    runner="pooling",
    enforce_eager=True,
    max_num_seqs=4,
    max_model_len=256,
    additional_config={
        "device_group": [0],
        "override_qaic_config": {
            "pooling_device": "cpu",
            "embed_seq_len": [32, 256],  # always include max_model_len
        },
    },
)
outputs = model.embed(["Hello, my name is"] * 10)

# QAIC pooling — single sequence length
model = LLM(
    model="intfloat/multilingual-e5-large",
    runner="pooling",
    enforce_eager=True,
    max_num_seqs=4,
    max_model_len=256,
    additional_config={
        "device_group": [0],
        "override_qaic_config": {
            "pooling_device": "qaic",
            "pooling_method": "mean",
            "normalize": True,
        },
    },
)
outputs = model.embed(["Hello, my name is"] * 10)

for prompt, output in zip(prompts, outputs):
    embeds = output.outputs.embedding
    print(f"Prompt: {prompt!r}, Embedding size: {len(embeds)}")
```

Run the full example:

```bash
python examples/offline_inference/basic/qaic_embed.py
```

## Configuration

| Parameter | Description |
|---|---|
| `runner` | Set to `"pooling"` for embedding models |
| `task` | `"embed"`, `"reward"`, `"classify"`, or `"score"` |
| `override_qaic_config.pooling_device` | `"qaic"` to run pooler on device, `"cpu"` to run on CPU |
| `override_qaic_config.pooling_method` | Pooling method for `qaic` device: `"mean"`, `"avg"`, `"cls"`, `"max"`, or custom |
| `override_qaic_config.normalize` | `True` to apply L2 normalization to pooled outputs (`qaic` only) |
| `override_qaic_config.softmax` | `True` to apply softmax to pooled outputs (`qaic` only) |
| `override_qaic_config.embed_seq_len` | List of sequence lengths to compile for, e.g. `[32, 256]`. Must include `max_model_len` |
| `override_pooler_config` | Pass a `PoolerConfig` object with `pooling_type`, `normalize`, and `softmax` |

## Notes

- Set `max_seq_len_to_capture` equal to the context length. For multi-sequence-length compilation, `max_model_len` must be one of the values in `embed_seq_len`.
- Use the correct API for the task: `embed()`, `encode()`, `classify()`, or `score()`.
- Jina models require `trust_remote_code=True`.

!!! warning "jina-embeddings-v2-base-en accuracy patch"
    This model requires a one-time patch for accuracy:

    ```python
    from QEfficient import QEFFAutoModel
    import os, subprocess, requests

    qeff_model = QEFFAutoModel.from_pretrained(
        "jinaai/jina-embeddings-v2-base-en", trust_remote_code=True
    )
    os.chdir(os.path.join(
        os.environ.get("HF_HOME"),
        "modules/transformers_modules/jinaai/jina-bert-implementation/"
        "f3ec4cf7de7e561007f27c9efc7148b0bd713f81/"
    ))
    response = requests.get(
        "https://huggingface.co/jinaai/jina-bert-implementation/discussions/7/files.diff"
    )
    with open("pr7.diff", "wb") as f:
        f.write(response.content)
    subprocess.run(["patch", "-p1", "-i", "pr7.diff"], check=True)
    ```

# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------

import time

import librosa

from vllm import LLM, SamplingParams
from vllm.utils.argparse_utils import FlexibleArgumentParser

# define qpc parameters
decode_bsz = 1
device_group = [0]

# encoder context length in tokens (Whisper mel-spectrogram frames)
encoder_ctx_len = 1500
# decoder context length in tokens
ctx_len = 150


def main(args):
    # Create a Whisper encoder/decoder model instance
    llm = LLM(
        model="openai/whisper-tiny.en",
        max_num_seqs=decode_bsz,  # determines decode batch size
        max_model_len=ctx_len,
        max_num_batched_tokens=encoder_ctx_len,
        quantization="mxfp6",  # Preferred quantization
        enable_prefix_caching=False,
        limit_mm_per_prompt={"audio": 1},
        hf_overrides={"max_source_positions": encoder_ctx_len},
        additional_config={"device_group": device_group},
    )

    audio = librosa.load(args.file_path, sr=None)

    # For whisper on qaic, only PL=1 is supported.
    # Also, continuous batching is not supported.
    prompt = {"prompt": "<|startoftranscript|>", "multi_modal_data": {"audio": audio}}

    # Create a sampling params object.
    sampling_params = SamplingParams(
        temperature=0,
        top_p=1.0,
        max_tokens=200,
    )

    start = time.time()

    # Generate output tokens from the prompts. The output is a list of
    # RequestOutput objects that contain the prompt, generated
    # text, and other information.
    outputs = llm.generate(prompt, sampling_params)

    # Print the outputs.
    for output in outputs:
        generated_text = output.outputs[0].text
        print(f"Generated text: {generated_text!r}")

    duration = time.time() - start

    print(f"Duration: {duration:.2f}s")


if __name__ == "__main__":
    parser = FlexibleArgumentParser(
        description="Demo on using vLLM for offline inference with "
        "whisper model for speech to text"
    )
    parser.add_argument(
        "--file-path",
        type=str,
        help="File path for local audio file, e.g. audio.wav",
    )

    args = parser.parse_args()
    if args.file_path is None:
        parser.print_help()
        raise ValueError("File-path must be provided.")
    main(args)

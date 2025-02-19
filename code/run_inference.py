from megatron.core.inference.common_inference_params import CommonInferenceParams
from nemo.collections.llm.recipes.precision.mixed_precision import bf16_mixed
from nemo import lightning as nl
import nemo_run as run
from nemo.collections import llm
import torch
import pytorch_lightning as pl
from pathlib import Path
from megatron.core.optimizer import OptimizerConfig
from nemo.collections.llm import Llama2Config7B
from typing import List, Optional
from nemo.lightning.io.mixin import IOMixin

input_data="/workspace/data/verilog/test.jsonl"
base_llama_path = "/root/.cache/nemo/models/Llama-2-7b-hf"
sft_ckpt_path = "/workspace/sft_log/checkpoints/sft_log--val_loss=1.3609-epoch=0-consumed_samples=20.0-last"
output_path_base="/workspace/inference/base_llama_prediction.jsonl"
output_path_sft="/workspace/inference/sft_prediction.jsonl"


def trainer() -> run.Config[nl.Trainer]:
    strategy = run.Config(
        nl.MegatronStrategy,
        tensor_model_parallel_size=2
    )
    trainer = run.Config(
        nl.Trainer,
        devices=2,
        max_steps=200,
        accelerator="gpu",
        strategy=strategy,
        plugins=bf16_mixed(),
        log_every_n_steps=20,
        limit_val_batches=2,
        val_check_interval=5,
        num_sanity_val_steps=0,
    )
    return trainer

# Configure inference to predict on base model checkpoint
def configure_inference_base():
    return run.Partial(
        llm.generate,
        path=str(base_llama_path),
        trainer=trainer(),
        input_dataset=input_data,
        inference_params=CommonInferenceParams(num_tokens_to_generate=50, top_k=1),
        output_path=output_path_base,
    )

# Configure inference to predict on trained DAPT checkpoint
def configure_inference_sft():
    return run.Partial(
        llm.generate,
        path=str(sft_ckpt_path),
        trainer=trainer(),
        input_dataset=input_data,
        inference_params=CommonInferenceParams(num_tokens_to_generate=50, top_k=1),
        output_path=output_path_sft,
    )

def local_executor_torchrun(nodes: int = 1, devices: int = 2) -> run.LocalExecutor:
    # Env vars for jobs are configured here
    env_vars = {
        "TORCH_NCCL_AVOID_RECORD_STREAMS": "1",
        "NCCL_NVLS_ENABLE": "0",
    }

    executor = run.LocalExecutor(ntasks_per_node=devices, launcher="torchrun", env_vars=env_vars)
    return executor

if __name__ == '__main__':
    print("running inference on base model")
    run.run(configure_inference_base(), executor=local_executor_torchrun())
    print("running inference on supervise fine tuned model")
    run.run(configure_inference_sft(), executor=local_executor_torchrun())
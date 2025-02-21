# +
# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# if compute uses slurm, request compute
#srun -G 8 --exclusive -p dgxa100_80g_2tb --pty -t 24:00:00 bash

docker run -it -p 8080:8080 -p 8088:8088 --rm --gpus '"device=0,1"' --ipc=host --network host -v $(pwd):/workspace nvcr.io/nvidia/nemo:24.12

python -m pip install --upgrade pip
source token.env
echo "installing huggingface hub"
pip3 install -U "huggingface_hub[cli]"
huggingface-cli login --token $HF_ACCESS_TOKEN

## download Llama2-7b model from HF and convert the format
mkdir Llama-2-7b/
echo "downloading llama2 7b model checkpoint into folder"
huggingface-cli download meta-llama/Llama-2-7b-hf --local-dir Llama-2-7b
echo "finished downloading Llama-2-7b model from huggingface"

echo "convert nemo model from .hf format to .nemo format, this will take a while..."
python3 /opt/NeMo/scripts/checkpoint_converters/convert_llama_hf_to_nemo.py --input_name_or_path=./Llama-2-7b/ --output_path=Llama-2-7b.nemo
# check if the converted file exist
if [ -f "Llama-2-7b.nemo" ]; then
    echo "model format conversion finished, delete huggingface model file"
    rm -rf Llama-2-7b/
else 
    echo "format conversion failed, exit"
    exit
fi

##### training script for actual sft
TRAINING_LOG_DIR="/results"
MODEL=Llama-2-7b.nemo
CONCAT_SAMPLING_PROBS="[1.0]"

# set tensor and pipeline parallel size
TP_SIZE=2
PP_SIZE=1

# key training parameters
MICRO_BATCH_SIZE=2
GLOBAL_BATCH_SIZE=4
LR=5e-6
MAX_STEP=100

# now run SFT command by appropriately setting the values for the parameters needed to run the job
echo "running supervised fine tuning step..."
torchrun --nproc_per_node=2 \
/opt/NeMo/examples/nlp/language_modeling/tuning/megatron_gpt_finetuning.py \
   trainer.precision=bf16 \
   trainer.devices=${TP_SIZE} \
   trainer.num_nodes=1 \
   trainer.val_check_interval=0.1 \
   trainer.max_steps=${MAX_STEP} \
   model.restore_from_path=${MODEL} \
   model.micro_batch_size=${MICRO_BATCH_SIZE} \
   model.global_batch_size=${GLOBAL_BATCH_SIZE} \
   model.tensor_model_parallel_size=${TP_SIZE} \
   model.pipeline_model_parallel_size=${PP_SIZE} \
   model.megatron_amp_O2=True \
   model.sequence_parallel=True \
   model.activations_checkpoint_granularity=selective \
   model.activations_checkpoint_method=uniform \
   model.optim.name=distributed_fused_adam \
   model.optim.lr=${LR} \
   model.answer_only_loss=True \
   model.peft.peft_scheme=none \
   model.data.train_ds.file_names=["code/data/merged/MG-Verilog_detailed_global_summary_in_out_train.jsonl"] \
   model.data.validation_ds.file_names=["code/data/merged/MG-Verilog_high_level_global_summary_in_out_validation.jsonl"] \
   model.data.test_ds.file_names=["code/data/merged/MG-Verilog_high_level_global_summary_in_out_test.jsonl"] \
   model.data.train_ds.concat_sampling_probabilities=${CONCAT_SAMPLING_PROBS} \
   model.data.train_ds.max_seq_length=2048 \
   model.data.validation_ds.max_seq_length=2048 \
   model.data.train_ds.micro_batch_size=${MICRO_BATCH_SIZE} \
   model.data.train_ds.global_batch_size=${GLOBAL_BATCH_SIZE} \
   model.data.validation_ds.micro_batch_size=${MICRO_BATCH_SIZE} \
   model.data.validation_ds.global_batch_size=${GLOBAL_BATCH_SIZE} \
   model.data.test_ds.micro_batch_size=${MICRO_BATCH_SIZE} \
   model.data.test_ds.global_batch_size=${GLOBAL_BATCH_SIZE} \
   model.data.train_ds.num_workers=${TP_SIZE} \
   model.data.validation_ds.num_workers=${TP_SIZE} \
   model.data.test_ds.num_workers=${TP_SIZE} \
   model.data.validation_ds.metric.name=loss \
   model.data.test_ds.metric.name=loss \
   exp_manager.create_wandb_logger=False \
   exp_manager.explicit_log_dir=${TRAINING_LOG_DIR} \
   exp_manager.resume_if_exists=True \
   exp_manager.resume_ignore_no_checkpoint=True \
   exp_manager.create_checkpoint_callback=True \
   exp_manager.checkpoint_callback_params.monitor=validation_loss \
   exp_manager.checkpoint_callback_params.save_best_model=True \
   exp_manager.checkpoint_callback_params.save_nemo_on_train_end=True \
   exp_manager.checkpoint_callback_params.mode=min \
   ++cluster_type=BCP

# following training and checkpoint saved, we run evaluation on original llama model
PATH_TO_TRAINED_MODEL=/results/checkpoints/megatron_gpt_peft_none_tuning.nemo

python3 /opt/NeMo/examples/nlp/language_modeling/tuning/megatron_gpt_generate.py \
    model.restore_from_path=${PATH_TO_TRAINED_MODEL} \
    trainer.devices=2 \
    model.micro_batch_size=${MICRO_BATCH_SIZE} \
    model.global_batch_size=${GLOBAL_BATCH_SIZE} \
    model.data.test_ds.file_names=["code/data/merged/MG-Verilog_high_level_global_summary_in_out_test.jsonl"] \
    model.data.test_ds.names=['llama_model'] \
    model.data.test_ds.global_batch_size=${GLOBAL_BATCH_SIZE} \
    model.data.test_ds.micro_batch_size=${MICRO_BATCH_SIZE} \
    model.data.test_ds.tokens_to_generate=2048 \
    model.tensor_model_parallel_size=${TP_SIZE} \
    model.pipeline_model_parallel_size=${PP_SIZE} \
    inference.greedy=True \
    model.data.test_ds.output_file_path_prefix=/results/sft_results \
    model.data.test_ds.write_predictions_to_file=True
#!/bin/bash
MODEL_PATH="/home/ubuntu/public/wdq_workspace/MedicalLLM/finetune/qwen_merged"
MODEL_NAME="qwen-1.5-7b-lora"
QUA_MODEL_PATH=""
QUA_MODEL_NAME=""

python -m vllm.entrypoints.openai.api_server \
    --host 127.0.0.1 \
    --port 8000 \
    --model $MODEL_PATH \
    --served-model-name $MODEL_NAME \
    --gpu-memory-utilization 0.6 \
    --max-model-len 4096 \
    --max-seq-len-to-capture 2048 \
    --tensor-parallel-size 2 \
    # --quantization gptq
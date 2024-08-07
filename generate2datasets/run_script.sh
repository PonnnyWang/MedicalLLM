#!/bin/bash

python generate_dialogues.py \
    --file_path "/root/LLMproject/MedicalLLM/data/TxtWithOutput/中国康复医学杂志_large.jsonl" \
    --save_path "/root/LLMproject/MedicalLLM/data/TxtWithOutput/output_dialogues_large.json" \
    --language "zh" \
    --instruction "general" \
    --assistant_word_count 300 \
    --human_word_count 50 \
    --num_turn_ratios 0.0 0.0 0.0 0.1 0.9
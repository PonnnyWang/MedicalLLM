#!/bin/bash

python generate_dialogues.py \
    --file_path "/root/LLMproject/MedicalLLM/data/中国康复医学杂志_test.json" \
    --save_path "/root/LLMproject/MedicalLLM/data/output_dialogues.json" \
    --language "zh" \
    --assistant_word_count 300 \
    --human_word_count 50 \
    --num_turn_ratios 0 0 0.2 0.5 0.3
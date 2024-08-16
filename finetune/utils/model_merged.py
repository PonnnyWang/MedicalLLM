def model_merged(mode_path, lora_path, new_model_directory):
    # 模型合并存储
    tokenizer = AutoTokenizer.from_pretrained(mode_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(mode_path, device_map="auto",torch_dtype=torch.float16, trust_remote_code=True).eval()
    model = PeftModel.from_pretrained(model, model_id=lora_path)
    merged_model = model.merge_and_unload()
    # 将权重保存为safetensors格式的权重, 且每个权重文件最大不超过2GB(2048MB)
    merged_model.save_pretrained(new_model_directory, max_shard_size="2048MB", safe_serialization=True)
    # 将mode_path中的tokenizer.json文件复制到new_model_directory目录下
    tokenizer_path = os.path.join(mode_path, 'tokenizer.json')
    new_tokenizer_path = os.path.join(new_model_directory, 'tokenizer.json')
    shutil.copyfile(tokenizer_path, new_tokenizer_path)
    print("new model has been merged!")

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch
"""
使用该脚本，将lora的权重合并大base model中
"""

def merge_lora_to_base_model():
    model_name_or_path = 'baichuan-inc/baichuan-7B'
    adapter_name_or_path = 'YeungNLP/firefly-baichuan-7b-qlora-sft'
    save_path = 'checkpoint/firefly-baichuan-7b-qlora-sft-merge'

    config = AutoConfig.from_pretrained(model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(
        adapter_name_or_path,
        trust_remote_code=True,
        # llama不支持fast
        use_fast=False if config.model_type == 'llama' else True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        # device_map='auto',
        device_map={'': 'cpu'}
    )
    model = PeftModel.from_pretrained(model, adapter_name_or_path, device_map={'': 'cpu'})
    model = model.merge_and_unload()

    tokenizer.save_pretrained(save_path)
    model.save_pretrained(save_path)


if __name__ == '__main__':
    merge_lora_to_base_model()

    
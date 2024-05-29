import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

peft_name = "./LoraModel"   #学習済みadapter_config.jsonのパス指定
output_dir = "./model"  #マージモデルの出力先

# PEFT(LoRA)の指定
peft_config = PeftConfig.from_pretrained(peft_name)
# ベースモデルの読み込み
model = AutoModelForCausalLM.from_pretrained(
    peft_config.base_model_name_or_path,
    return_dict=True,
    torch_dtype=torch.bfloat16,
)
# Rinnaのトークナイザーでは、「use_fast=False」も必要になる
tokenizer = AutoTokenizer.from_pretrained(peft_config.base_model_name_or_path,use_fast=False)
# PEFT(LoRA)の読み込み
model = PeftModel.from_pretrained(model, peft_name)
# マージモデル作成
merged_model = model.merge_and_unload()
# 出力
merged_model.save_pretrained(output_dir)  
tokenizer.save_pretrained(output_dir)
print(f"Saving to {output_dir}")  
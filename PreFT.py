## deepspeed --num_gpus=8 PreFT.py --deepspeed --deepspeed_config zero_train.json
from datasets import load_dataset
import time
import torch
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments,DataCollatorForLanguageModeling
import json
from peft import LoraConfig
import wandb

# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

# 以下処理を記入

wandb.login()

# まずCUDAが利用可能かどうかを確認
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dolly_dataset = load_dataset("neody/aozora-cleaned") #使いたいデータセットを入力

# 簡易化のためinputの値が空のデータに絞る
dolly_train_dataset = dolly_dataset['train']

model_name = "NTQAI/chatntq-ja-7b-v1.0"#使いたいモデルを選択

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
 model_name, torch_dtype=torch.bfloat16
)

# モデルをCUDAデバイスに移動
# model.to(device)

print(tokenizer.eos_token)
tokenizer.pad_token = tokenizer.unk_token

def formatting_prompts_func(example):
    return example['text']  # データセットに合わせてください


collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # マスク言語モデリング
)
# バッチサイズを減少させ、勾配累積のステップを調整
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    # LoRAの学習対象の層の指定　←　ここ
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
        "lm_head"
        ],
    )

args = TrainingArguments(
    output_dir='./output_lora',
    num_train_epochs=3,
    gradient_accumulation_steps=1,
    per_device_train_batch_size=1,
    save_strategy="no",
    logging_steps=20,
    lr_scheduler_type="constant",
    save_total_limit=1,
    bf16=True,
    report_to="wandb"
)


trainer = SFTTrainer(
    model,
    args=args,
    train_dataset=dolly_train_dataset,
    formatting_func=formatting_prompts_func,
    max_seq_length=1024,
    data_collator=collator,
    peft_config=peft_config,
)

trainer.train()
trainer.save_model("./LoraModel2")
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import json, torch
from datasets import Dataset
from peft import LoraConfig
from trl import DPOTrainer, DPOConfig
from transformers import AutoTokenizer, AutoModelForCausalLM

MID = "HuggingFaceTB/SmolLM2-360M-Instruct"
raw = json.load(open("data/persona_lora/ramsay_style_preferences.json"))
rows = [{"prompt": [{"role": "user", "content": r["prompt"]}],
         "chosen": [{"role": "assistant", "content": r["chosen"]}],
         "rejected": [{"role": "assistant", "content": r["rejected"]}]} for r in raw]
ds = Dataset.from_list(rows).train_test_split(test_size=0.1, seed=42)
tok = AutoTokenizer.from_pretrained(MID)
model = AutoModelForCausalLM.from_pretrained(MID, dtype=torch.float32)
peft_cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"])
cfg = DPOConfig(output_dir="outputs/lora_persona4", beta=0.1, learning_rate=1e-4,
    loss_type=["sigmoid","sft"], loss_weights=[1.0,1.0],
    per_device_train_batch_size=4, gradient_accumulation_steps=4,
    num_train_epochs=3, lr_scheduler_type="cosine", warmup_ratio=0.1,
    max_length=512, fp16=True, logging_steps=2,
    eval_strategy="epoch", save_strategy="no", report_to="none", seed=42)
trainer = DPOTrainer(model=model, args=cfg, train_dataset=ds["train"],
    eval_dataset=ds["test"], processing_class=tok, peft_config=peft_cfg)
trainer.train()
trainer.save_model("outputs/lora_persona4/adapter")
print("saved")

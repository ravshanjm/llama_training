import os
import torch
import wandb
from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import ORPOConfig, ORPOTrainer, setup_chat_format

wandb.login(key='190ea355e042b66c717ec2994563d1e8cf420446')

def main():
    attn_implementation = "flash_attention_2"
    torch_dtype = torch.bfloat16

    # Model
    base_model = "meta-llama/Meta-Llama-3-8B"
    new_model = "OrpoLlama-3-8B"

    # QLoRA config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch_dtype,
        bnb_4bit_use_double_quant=True,
    )

    # LoRA config
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=['up_proj', 'down_proj', 'gate_proj', 'k_proj', 'q_proj', 'v_proj', 'o_proj']
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    def format_chat_template(row):
        row["chosen"] = tokenizer.apply_chat_template(row["chosen"], tokenize=False)
        row["rejected"] = tokenizer.apply_chat_template(row["rejected"], tokenize=False)
        return row

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
        attn_implementation=attn_implementation
    )
    model, tokenizer = setup_chat_format(model, tokenizer)
    model = prepare_model_for_kbit_training(model)
    dataset = load_dataset("aisha-org/orpo_dataset_v1")

    dataset = dataset['train'].shuffle(seed=42).select(range(200))

    dataset = dataset.map(
        format_chat_template,
        num_proc=os.cpu_count(),
    )

    dataset = dataset.train_test_split(test_size=0.02)

    # DeepSpeed configuration
    deepspeed_config = {
        "fp16": {
            "enabled": "auto",
            "loss_scale": 0,
            "loss_scale_window": 1000,
            "initial_scale_power": 16,
            "hysteresis": 2,
            "min_loss_scale": 1
        },
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": "auto",
                "betas": "auto",
                "eps": "auto",
                "weight_decay": "auto"
            }
        },
        "scheduler": {
            "type": "WarmupLR",
            "params": {
                "warmup_min_lr": "auto",
                "warmup_max_lr": "auto",
                "warmup_num_steps": "auto"
            }
        },
        "zero_optimization": {
            "stage": 3,
            "offload_optimizer": {
                "device": "cpu",
                "pin_memory": True
            },
            "offload_param": {
                "device": "cpu",
                "pin_memory": True
            },
            "overlap_comm": True,
            "contiguous_gradients": True,
            "sub_group_size": 1e9,
            "reduce_bucket_size": "auto",
            "stage3_prefetch_bucket_size": "auto",
            "stage3_param_persistence_threshold": "auto",
            "stage3_max_live_parameters": 1e9,
            "stage3_max_reuse_distance": 1e9,
            "stage3_gather_fp16_weights_on_model_save": True
        },
        "gradient_accumulation_steps": "auto",
        "gradient_clipping": "auto",
        "steps_per_print": 2000,
        "train_batch_size": "auto",
        "train_micro_batch_size_per_gpu": "auto",
        "wall_clock_breakdown": False
    }

    orpo_args = ORPOConfig(
        learning_rate=9e-6,
        beta=0.1,
        lr_scheduler_type="linear",
        max_length=1024,
        max_prompt_length=1024,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        optim="paged_adamw_8bit",
        num_train_epochs=1,
        evaluation_strategy="steps",
        eval_steps=0.2,
        logging_steps=1,
        warmup_steps=10,
        report_to="wandb",
        output_dir="./results",
        deepspeed=deepspeed_config,  # Add DeepSpeed configuration
    )

    trainer = ORPOTrainer(
        model=model,
        args=orpo_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        peft_config=peft_config,
        tokenizer=tokenizer,
    )
    trainer.train()
    trainer.save_model(new_model)

if __name__ == "__main__":
    main()

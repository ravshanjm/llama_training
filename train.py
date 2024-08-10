import os
import gc
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
from accelerate import Accelerator
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
    CPUOffload,
)
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    enable_wrap,
    wrap,
)

wandb.login(key='190ea355e042b66c717ec2994563d1e8cf420446')

def main():
    accelerator = Accelerator()
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

    # FSDP settings
    def create_fsdp_policy():
        return size_based_auto_wrap_policy(min_num_params=1e8)

    fsdp_config = dict(
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=MixedPrecision(
            param_dtype=torch.float16,
            reduce_dtype=torch.float16,
            buffer_dtype=torch.float16,
        ),
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        cpu_offload=CPUOffload(offload_params=True),
        auto_wrap_policy=create_fsdp_policy,
    )

    with accelerator.main_process_first():
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=bnb_config,
            device_map="auto",
            attn_implementation=attn_implementation
        )
        model, tokenizer = setup_chat_format(model, tokenizer)
        model = prepare_model_for_kbit_training(model)

    # Wrap the model with FSDP
    model = FSDP(model, **fsdp_config)

    dataset = load_dataset("aisha-org/orpo_dataset_v1")
    dataset = dataset['train'].shuffle(seed=42).select(range(200))

    with accelerator.main_process_first():
        dataset = dataset.map(
            format_chat_template,
            num_proc=os.cpu_count(),
        )
        dataset = dataset.train_test_split(test_size=0.02)

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
        fsdp=fsdp_config,
        fsdp_transformer_layer_cls_to_wrap="LlamaDecoderLayer",
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
    
    # Save the model on the main process
    if accelerator.is_main_process:
        trainer.save_model(new_model)

if __name__ == "__main__":
    main()

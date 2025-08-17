import argparse
import os
import json
from dataclasses import dataclass, asdict

import torch

from transformers import (
    TrainingArguments,
    EarlyStoppingCallback,
)

from unsloth import is_bfloat16_supported
from trl import SFTTrainer

from toc_llm.utils import get_experiment_id, set_seed, load_json_data
from toc_llm.data.wiki_727 import load_datasets as load_datasets_wiki_727
from toc_llm.training import (
    TrainingConfigBase, 
    determine_assistant_start_token, 
    load_tokenizer_n_model, 
    DataCollator, 
    save_model,
    infer_on_test_set,
)
from toc_llm.inferrer import TocLlmInferrer, TocLlmInferrerConfig
from toc_llm.eval import Metrics, merge_metrics


@dataclass
class TrainingConfigWiki727(TrainingConfigBase):
    wiki_727_dir: str = ""  # Directory to wiki-727 dataset
    wiki_727_max_tokens: int | None = None  # Currently not used, always set to 64000 max tokens
    max_samples: int | None = None  # Number of samples to load from each wiki-727 set (train/dev/test)
    run_test: bool = False  # Whether to run evaluation on the test set after training


def run_training(config: TrainingConfigWiki727):
    set_seed(config.seed)

    tokenizer, model = load_tokenizer_n_model(config)
    approx_max_num_tokens_transcript = tokenizer.model_max_length - 7500  # reserve 7500 for prompt and toc
    if config.wiki_727_max_tokens is not None:
        max_input_text_tokens = min(config.wiki_727_max_tokens, approx_max_num_tokens_transcript)
    else:
        max_input_text_tokens = None

    # Load data from wiki-727
    train_dataset, _, test_dataset = load_datasets_wiki_727(
        dataset_root=config.wiki_727_dir,
        model_id=config.model_id,
        max_seq_len=max_input_text_tokens,
        max_samples=config.max_samples,
        load_dev=False,
        load_test=config.run_test,
    )
    start_token_of_assistant = determine_assistant_start_token(config.model_id)
    start_assistant_id = tokenizer.encode(start_token_of_assistant, add_special_tokens=False)[0]
    assistant_tokens_mask = "{% generation %}" in tokenizer.chat_template
    data_collator = DataCollator(tokenizer, config, assistant_tokens_mask, start_assistant_id, add_pause=False)

    if config.from_checkpoint is not None:
        experiment_id = config.from_checkpoint
    else:
        experiment_id = get_experiment_id(prefix="train")
    output_dir = str(os.path.join(config.output_dir, experiment_id))
    if config.from_checkpoint is None:
        os.makedirs(output_dir, exist_ok=False)
    print(f"Output directory: {output_dir}\n", flush=True)
    gradient_accumulation_steps = config.accumulate_steps if config.accumulate_steps else 1 

    if config.warmup_steps is not None:
        warmup_steps = config.warmup_steps
    elif config.max_samples is None:
        warmup_steps = 100
    else:
        num_steps = config.max_samples / (config.batch_size * gradient_accumulation_steps)
        num_steps = int(num_steps + 0.5)  # round up
        warmup_steps = int(num_steps / 10)  # use 10% of steps for warmup
        warmup_steps = min(warmup_steps, 100)  # use max 100 warmup steps
    training_args = TrainingArguments(
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        save_strategy="steps",
        save_steps=config.save_steps,
        save_total_limit=3,  # Only keep last 3 checkpoints
        learning_rate=config.learning_rate,
        warmup_steps=warmup_steps,
        weight_decay=0.01,
        num_train_epochs=1,
        max_steps=config.max_steps if config.max_steps is not None else -1,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=config.logging_steps,
        output_dir=output_dir,
        optim="adamw_8bit",
        seed=config.seed,
        report_to="none",
    )
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        args=training_args,
        output_dir=output_dir,
        data_collator=data_collator,
        callbacks=None,
    )
    if config.from_checkpoint is not None:
        print(f"\nResume training from checkpoint:{config.from_checkpoint}\n", flush=True)
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    save_model(trainer, model, tokenizer, output_dir)

    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, "w") as outp:
        json.dump(asdict(config), outp, indent=2)
    
    if test_dataset is not None:
        print("\nEvaluating on test set...", flush=True)
        empty_config = TocLlmInferrerConfig(checkpoint="")
        inferrer = TocLlmInferrer(empty_config)
        inferrer.set_model_and_tokenizer(model, tokenizer, add_pause=False)
        test_results, test_metrics = infer_on_test_set(inferrer, test_dataset)
        test_metrics_avg = merge_metrics(test_metrics, use_weighted_avg=False)
        print("\nTest metrics:", flush=True)
        for m, val in test_metrics_avg.items():
            print(f"- {m}: {round(val*100, 2)}")


def main():
    max_split_size = 256
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = f"max_split_size_mb:{max_split_size}"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-file", type=str, required=True,
        help="Path to the training config file (JSON). Must include the path to the wiki-727 dataset."
    )
    args = parser.parse_args()
    assert os.path.isfile(args.config_file), "Provided config file does not exist."
    print(f"Using settings from config file: {args.config_file}", flush=True)
    custom_config = load_json_data(args.config_file)
    config = TrainingConfigWiki727(**custom_config)
    run_training(config)


if __name__ == "__main__":
    main()

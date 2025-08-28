import argparse
import os
import json
from dataclasses import dataclass, asdict

from transformers import (
    TrainingArguments,
    EarlyStoppingCallback,
)
from unsloth import is_bfloat16_supported
from trl import SFTTrainer


from toc_llm.utils import get_experiment_id, set_seed, load_json_data
from toc_llm.training import (
    TrainingConfigBase, 
    determine_assistant_start_token, 
    load_tokenizer_n_model, 
    DataCollator, 
    save_model,
    infer_on_test_set,
)
from toc_llm.inferrer import TocLlmInferrer, TocLlmInferrerConfig
from toc_llm.eval import merge_metrics


def load_datasets_custom():
    """
    Load torch datasets for custom data.
    Should return train, dev, and test datasets.
    """
    raise NotImplementedError("This function should be implemented to load custom datasets."
                              "See also `toc_llm/data/toc.py` for more information on the expected dataset.")


@dataclass
class TrainingConfigCustom(TrainingConfigBase):
    add_pause: bool = False


def run_training(config: TrainingConfigCustom):
    set_seed(config.seed)

    tokenizer, model = load_tokenizer_n_model(config)
    
    # Load data from custom datasets (to be implemented)
    train_dataset, dev_dataset, test_dataset = load_datasets_custom()

    start_token_of_assistant = determine_assistant_start_token(config.model_id)
    start_assistant_id = tokenizer.encode(start_token_of_assistant, add_special_tokens=False)[0]
    assistant_tokens_mask = "{% generation %}" in tokenizer.chat_template
    data_collator = DataCollator(tokenizer, config, assistant_tokens_mask, start_assistant_id, add_pause=False)
    # Training arguments
    if config.from_checkpoint is not None:
        experiment_id = config.from_checkpoint
    else:
        experiment_id = get_experiment_id()
    output_dir = str(os.path.join(config.output_dir, experiment_id))
    if config.from_checkpoint is None:
        os.makedirs(output_dir, exist_ok=False)
    print(f"Output directory: {output_dir}\n", flush=True)
    gradient_accumulation_steps = config.accumulate_steps if config.accumulate_steps else 1

    if config.warmup_steps is not None:
        warmup_steps = config.warmup_steps
    else:
        num_steps = len(train_dataset) / (config.batch_size * gradient_accumulation_steps)
        num_steps = int(num_steps + 0.5)  # round up
        warmup_steps = int(num_steps / 10)  # use 10% of first epoch steps for warmup
    training_args = TrainingArguments(
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        save_strategy="epoch",
        eval_strategy="epoch",
        save_steps=config.save_steps,
        load_best_model_at_end=True,
        save_total_limit=3,  # Only keep last 3 checkpoints
        dataloader_num_workers=8,
        learning_rate=config.learning_rate,
        warmup_steps=warmup_steps,
        weight_decay=0.01,
        num_train_epochs=config.max_epochs,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=config.logging_steps,
        output_dir=output_dir,
        optim="adamw_8bit",
        seed=config.seed,
        report_to="none",
    )
    callbacks = [
        EarlyStoppingCallback(early_stopping_patience=config.patience)
    ]
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        tokenizer=tokenizer,
        args=training_args,
        output_dir=output_dir,
        data_collator=data_collator,
        callbacks=callbacks,
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
        help="Path to the training config file (JSON)."
    )
    args = parser.parse_args()
    assert os.path.isfile(args.config_file), "Provided config file does not exist."
    print(f"Using settings from config file: {args.config_file}", flush=True)
    custom_config = load_json_data(args.config_file)
    config = TrainingConfigCustom(**custom_config)
    run_training(config)


if __name__ == "__main__":
    main()

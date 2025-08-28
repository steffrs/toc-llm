import os
import json
from dataclasses import dataclass

from unsloth import FastLanguageModel
import torch

from .utils import count_parameters, load_json_data, get_linear_segmentation
from .inferrer import TocLlmInferrer
from .data.toc import (
    create_messages,
    toc_to_label_dict,
)
from .eval import hierarchical_metric, Metrics


@dataclass
class BaseConfig:
    model_id: str = "unsloth/Mistral-Nemo-Instruct-2407-bnb-4bit"
    output_dir: str = "./train_output"
    seed: int = 0



@dataclass
class TrainingConfigBase:
    output_dir: str = "./train_output"
    batch_size: int = 1
    accumulate_steps: int = 8  # if None, update after every batch
    learning_rate: float = 5e-5
    warmup_steps: int | None = None  # if None, use around 10% of steps for warmup
    # regularization: float = 0.001

    model_id: str = "unsloth/Mistral-Nemo-Instruct-2407-bnb-4bit"  # or .../final-lora-only directory path
    rank: int = 16
    lora_dropout: float = 0.05
    lora_alpha: int = 32  # Hyperparam: often 16–64
    max_length: int | None = None  # if None, default max_length of model will be used

    max_steps: int | None = None
    max_epochs: int | None = None
    patience: int = 3
    logging_steps: int = 200
    save_steps: int = 200
    from_checkpoint: str | None = None
    seed: int = 0


class DataCollator:
    """
    Data collator for preparing batches of data for training.
    It handles the tokenization of messages, manages assistant tokens, and prepares labels for training.
    
    :param tokenizer: Tokenizer to use for tokenizing messages.
    :param config: Training configuration containing model parameters.
    :param assistant_tokens_mask: Whether to mask assistant tokens in the labels.
    :param start_assistant_id: The token ID that marks the start of the assistant's response.
    :param add_pause: Whether to add pause durations to the transcript sentences.
    """

    def __init__(self, tokenizer, config: TrainingConfigBase,
                 assistant_tokens_mask: bool, start_assistant_id: int, add_pause: bool):
        self.tokenizer = tokenizer
        self.model_max_length = tokenizer.model_max_length
        self.config = config
        self.assistant_tokens_mask = assistant_tokens_mask
        self.start_assistant_id = start_assistant_id
        self.is_mistral = "mistral" in config.model_id.lower()
        self.add_pause = add_pause

    def __call__(self, batch):
        samples = [create_messages(
            b[0], b[1], self.is_mistral, add_pause=self.add_pause
        ) for b in batch]
        tokenized = self.tokenizer.apply_chat_template(
            conversation=samples,
            tokenize=True,
            truncation=True,
            max_length=self.config.max_length,
            padding="longest",
            return_tensors="pt",
            add_generation_prompt=False,
            return_dict=True,
            return_assistant_tokens_mask=self.assistant_tokens_mask,
        )
        labels = tokenized["input_ids"].clone()
        if self.assistant_tokens_mask:
            for i in range(len(samples)):  # iterate over batch samples
                assistant_mask = tokenized["assistant_tokens_mask"][i]
                labels[i][~assistant_mask] = -100
        else:
            for i in range(len(samples)):
                input_ids = tokenized["input_ids"][i].tolist()
                start_assistant_index = [
                    idx for idx, ident in enumerate(input_ids) if ident == self.start_assistant_id
                ][-1]
                labels[i, :start_assistant_index] = -100
        tokenized["labels"] = labels
        return tokenized


def load_tokenizer_n_model(config: TrainingConfigBase):
    """Load the tokenizer and model based on the provided configuration."""
    print(f"Use tokenizer/model ID: '{config.model_id}'", flush=True)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.model_id,
        dtype=None,
        load_in_4bit=True
    )
    if "llama" in config.model_id.lower():
        tokenizer.pad_token_id = tokenizer.eos_token_id  # Llama often doesn't have a pad_token, reuse eos

    model = FastLanguageModel.get_peft_model(
        model,
        r=config.rank,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=config.seed,
    )
    count_parameters(model)
    return tokenizer, model


def determine_assistant_start_token(model_id: str) -> str:
    """Determine the start token for the assistant based on the model ID."""
    if os.path.isdir(model_id):  # pre-trained, locally saved model
        adapter_config_path = os.path.join(model_id, "adapter_config.json")
        config = load_json_data(adapter_config_path)
        model_id = config["base_model_name_or_path"]
    if "phi" in model_id.lower():  # unsloth/Phi-3.5-mini-instruct-bnb-4bit
        if "3.5" in model_id:
            return "<|assistant|>"
        else:
            return "<|im_start|>"
    if "llama" in model_id.lower():  # unsloth/Llama-3.2-3B-Instruct-bnb-4bit
        return "<|start_header_id|>"
    if "qwen" in model_id.lower():  # unsloth/Qwen2.5-7B-Instruct-bnb-4bit
        return "<|im_start|>"
    if "mistral" in model_id.lower():
        return "[/INST]"
    raise ValueError("Unknown model ID, cannot determine assistant start token.")


def save_model(trainer, model, tokenizer, output_dir: str):
    """
    Save the final model and tokenizer to the output directory.
    This function saves the model in two ways:
    1. As a full model with LoRA weights and base model config.
    2. As a LoRA adapter only, which can be used for inference with the base model.

    :param trainer: SFTTrainer instance used for training.
    :param model: trained model instance.
    :param tokenizer: tokenizer instance used for the model.
    :param output_dir: directory where the model should be saved.
    """
    model_save_path = os.path.join(output_dir, "final-model")
    trainer.save_model(model_save_path)  # Saves adapter + base
    # Saving original model config
    config_path = os.path.join(model_save_path, "config.json")
    with open(config_path, "w") as outp:
        json.dump(model.config.to_dict(), outp, indent=2)
    # Saving tokenizer to model_save_path
    tokenizer.save_pretrained(model_save_path)

    # Saving LoRA adapter separately
    lora_save_path = os.path.join(output_dir, "final-lora-only")
    model.save_pretrained(lora_save_path)
    print("\nModel saved to:", model_save_path, flush=True)


def infer_on_test_set(inferrer: TocLlmInferrer, test_dataset: torch.utils.data.Dataset):
    """Run inference on the test dataset and calculate metrics."""
    all_samples = []
    all_metrics = []
    all_hier_metrics = []
    for transcript, ref_toc in test_dataset:
        hyp_toc = inferrer.infer_from_dataset_sample(transcript)
        num_sentences = len([ln for ln in transcript.split("\n") if ln.strip()])
        hyp = toc_to_label_dict(hyp_toc, num_sentences)
        ref = toc_to_label_dict(ref_toc, num_sentences)
        sample = {"hyp_seg": hyp, "ref_seg": ref, "name": "unknown", "hyp_toc": hyp_toc, "ref_toc": ref_toc}
        all_samples.append(sample)
        try:
            if not hyp:
                # If no segmentation hypothesis was created, use all 0s on level 1 as fallback (no segmentation)
                hyp = {1: [0 for _ in range(len(ref[1]))]}
            hyp_lin, _ = get_linear_segmentation(hyp, transform_to_change_labels=True)
            ref_lin, _ = get_linear_segmentation(ref, transform_to_change_labels=True)
            metrics = Metrics(hyp_lin, ref_lin, start_labels=False, add_segeval_metrics=True)
            all_metrics.append(metrics)
            hier_metric = hierarchical_metric(hyp, ref)
            all_hier_metrics.append(hier_metric)
        except Exception as e:
            print(f"Error during calculating metrics: {e}", flush=True)
    return all_samples, all_metrics

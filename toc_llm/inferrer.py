from dataclasses import dataclass
import os
import gc

import torch
from transformers import AutoTokenizer
from unsloth import FastLanguageModel

from toc_llm.utils import load_json_data
from toc_llm.data.toc import (
    create_messages,
    TableOfContents,
    sentences_to_llm_format,
    messages_to_tokenized,
    toc_to_label_dict,
)


@dataclass
class TocLlmInferrerConfig:
    checkpoint: str = ""  # Path to checkpoint path including adapters and config
    max_new_tokens: int = 1024  # Maximum number of new tokens to generate (used for table-of-contents)
    add_pause: bool | None = None  # Only used for zero- and few-shot settings


class TocLlmInferrer:

    def __init__(self, config: TocLlmInferrerConfig):
        self.cfg = config
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        if not self.cfg.checkpoint:
            # Case 1: Initialize empty inferrer without model (need to set model and tokenizer later)
            print("WARNING: Inferrer is not initialized with a checkpoint path, "
                  "make sure to set model and tokenizer using `set_model_and_tokenizer()` method.",
                  flush=True)
            return
        if os.path.isdir(self.cfg.checkpoint):
            # Case 2: Load from local directory with pre-trained adapters
            train_config_path = os.path.join(self.cfg.checkpoint, "config.json")
            self.train_config = load_json_data(train_config_path)
            print(f"Load pretrained adapters from: '{self.cfg.checkpoint}'", flush=True)
            self._load_pretrained_model()
            self._add_pause = self.train_config.get("add_pause", False)
        else:
            # Case 3: Load from Hugging Face model hub (Zero-shot or few-shot setting)
            print(f"Load huggingface tokenizer/model: '{self.cfg.checkpoint}'", flush=True)
            self._model, self._tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.cfg.checkpoint,
                load_in_4bit=True
            )
            FastLanguageModel.for_inference(self._model)
            self.is_mistral = "mistral" in self._tokenizer.name_or_path.lower()
            self._add_pause = config.add_pause if config.add_pause is not None else False

    def _load_pretrained_model(self):
        model_id = self.train_config["model_id"]
        self._tokenizer = AutoTokenizer.from_pretrained(model_id)
        model_save_path = os.path.join(self.cfg.checkpoint, "final-lora-only")
        self._model, self._tokenizer = FastLanguageModel.from_pretrained(
            model_save_path,
            load_in_4bit=True,
        )
        FastLanguageModel.for_inference(self._model)
        self.is_mistral = "mistral" in self._tokenizer.name_or_path.lower()

    def set_model_and_tokenizer(self, model, tokenizer, add_pause: bool = False):
        self._model = model
        FastLanguageModel.for_inference(self._model)
        self._tokenizer = tokenizer
        self.is_mistral = "mistral" in self._tokenizer.name_or_path.lower()
        self._add_pause = add_pause

    def __call__(self, sentences: list[str],
                 pauses: list[float] | None = None,
                 few_shot_samples: list[tuple[str, str]] | None = None,
                 ) -> TableOfContents | dict:
        transcript = sentences_to_llm_format(sentences, pauses=pauses)
        add_pause = pauses is not None
        if self._add_pause and (not add_pause):
            print("WARNING: Model was trained with pause information, but no pauses are provided.", flush=True)
        if (not self._add_pause) and add_pause:
            print("WARNING: Model was trained without pause information, but pauses are provided.", flush=True)
        messages = create_messages(transcript, toc=None, is_mistral=self.is_mistral,
                                   few_shot_samples=few_shot_samples, add_pause=add_pause)
        tokenized = messages_to_tokenized(messages, self._tokenizer, max_length=None, return_assistant_tm=False)
        gen_toc = self.run_on_samples(tokenized)
        segmentation = toc_to_label_dict(gen_toc, len(sentences))
        return gen_toc, segmentation

    def infer_from_dataset_sample(self, transcript: str) -> str:
        messages = create_messages(transcript, toc=None, is_mistral=self.is_mistral,
                                   add_pause=self._add_pause)
        tokenized = messages_to_tokenized(messages, self._tokenizer, max_length=None, return_assistant_tm=False)
        gen_toc = self.run_on_samples(tokenized)
        return gen_toc

    def run_on_samples(self, tokenized) -> str:
        with torch.no_grad():
            tokenized = tokenized.to(self._device)
            output = self._model.generate(**tokenized, max_new_tokens=self.cfg.max_new_tokens, do_sample=False)
        generated = self._tokenizer.decode(output[0], skip_special_tokens=True)
        if "message.assistant" in generated:
            gen_assistant = generated.split("message.assistant")[-1].strip()
        elif "system message." in generated:
            gen_assistant = generated.split("system message.")[-1].strip()
            gen_assistant = gen_assistant.lstrip("assistant").strip()
        tokenized = tokenized.to("cpu")
        del tokenized
        output = output.to("cpu")
        del output
        gc.collect()
        return gen_assistant

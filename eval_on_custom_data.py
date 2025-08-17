import argparse
import os
from dataclasses import dataclass

import torch

from toc_llm.utils import get_experiment_id, set_seed, load_json_data
from toc_llm.data.wiki_727 import load_datasets as load_datasets_wiki_727
from toc_llm.training import (
    load_tokenizer_n_model, infer_on_test_set
)
from toc_llm.inferrer import TocLlmInferrer, TocLlmInferrerConfig
from toc_llm.eval import merge_metrics, bootstrap_metrics


def load_test_dataset_custom() -> torch.utils.data.Dataset:
    """
    Load torch dataset for custom test data.
    Should return train, dev, and test datasets.
    """
    raise NotImplementedError("This function should be implemented to load a custom test dataset."
                              "See also `toc_llm/toc.py` for more information on the expected dataset.")


@dataclass
class EvalConfigCustomData:
    model_id: str = "unsloth/Mistral-Nemo-Instruct-2407-bnb-4bit"  # Original model ID used for training
    checkpoint: str = ""  # Path to the checkpoint to evaluate
    output_dir: str = "./eval_output"
    bootstrap_metrics: bool = False


def run_evaluation(config: EvalConfigCustomData):
    set_seed(config.seed)

    tokenizer, model = load_tokenizer_n_model(config)
    approx_max_num_tokens_transcript = tokenizer.model_max_length - 7500  # reserve 7500 for prompt and toc
    if config.wiki_727_max_tokens is not None:
        max_input_text_tokens = min(config.wiki_727_max_tokens, approx_max_num_tokens_transcript)
    else:
        max_input_text_tokens = None

    _, _, test_dataset = load_datasets_wiki_727(
        dataset_root=config.wiki_727_dir,
        model_id=config.model_id,
        max_seq_len=max_input_text_tokens,
        max_samples=config.max_samples,
        load_only_train=False,
    )
    experiment_id = get_experiment_id()
    experiment_id = f"eval-{experiment_id}"
    output_dir = str(os.path.join(config.output_dir, experiment_id))
    print(f"Output directory: {output_dir}\n", flush=True)

    empty_config = TocLlmInferrerConfig(checkpoint="")
    inferrer = TocLlmInferrer(empty_config)
    inferrer.set_model_and_tokenizer(model, tokenizer, add_pause=False)
    test_results, test_metrics = infer_on_test_set(inferrer, test_dataset)
    if config.bootstrap_metrics:
        print("Bootstrapping metrics...", flush=True)
        aggregated_metrics = bootstrap_metrics(test_metrics, round_values=5, use_weighted_avg=False)
    else:
        aggregated_metrics = merge_metrics(test_metrics, round_values=5, use_weighted_avg=False)

    return test_results, test_metrics


def main():
    max_split_size = 256
    # torch.multiprocessing.set_start_method('spawn', force=True)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = f"max_split_size_mb:{max_split_size}"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # torch.multiprocessing.set_sharing_strategy('file_system')

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-file", type=str, required=True,
        help="Path to the training config file (JSON). "
             "Must include the path to the model checkpoint."
    )
    args = parser.parse_args()
    assert os.path.isfile(args.config_file), "Provided config file does not exist."
    print(f"Using settings from config file: {args.config_file}", flush=True)
    custom_config = load_json_data(args.config_file)
    config = EvalConfigCustomData(**custom_config)
    run_evaluation(config)


if __name__ == "__main__":
    main()

import argparse
import os
from dataclasses import dataclass

import torch

from toc_llm.utils import get_experiment_id, set_seed, load_json_data
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

    test_dataset = load_test_dataset_custom()
    experiment_id = get_experiment_id()
    experiment_id = f"eval-{experiment_id}"
    output_dir = str(os.path.join(config.output_dir, experiment_id))
    print(f"Output directory: {output_dir}\n", flush=True)

    empty_config = TocLlmInferrerConfig(checkpoint="")
    inferrer = TocLlmInferrer(empty_config)
    inferrer.set_model_and_tokenizer(model, tokenizer, add_pause=False)
    test_results, test_metrics = infer_on_test_set(inferrer, test_dataset)
    # Save/post-process results as needed


def main():
    max_split_size = 256
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = f"max_split_size_mb:{max_split_size}"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

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

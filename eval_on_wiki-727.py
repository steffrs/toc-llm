import argparse
import os
import json
from dataclasses import dataclass

from toc_llm.utils import get_experiment_id, set_seed, load_json_data
from toc_llm.data.wiki_727 import load_datasets as load_datasets_wiki_727
from toc_llm.training import infer_on_test_set
from toc_llm.inferrer import TocLlmInferrer, TocLlmInferrerConfig
from toc_llm.eval import merge_metrics, bootstrap_metrics, aggregate_metrics_with_stddev


@dataclass
class EvalConfigWiki727:
    wiki_727_dir: str = ""  # Directory to wiki-727 dataset
    wiki_727_max_tokens: int | None = None  # Currently not used, always set to 64000 max tokens
    max_samples: int | None = None  # Number of samples to load from  wiki-727 test set
    model_id: str = "unsloth/Mistral-Nemo-Instruct-2407-bnb-4bit"  # Original model ID used for training
    checkpoint: str = ""  # Path to the checkpoint to evaluate
    output_dir: str = "./eval_output"
    bootstrap_metrics: bool = False
    num_bootstrap_samples: int = 100  # Number of bootstrap samples to use for metrics aggregation
    seed: int = 0


def run_evaluation(config: EvalConfigWiki727):
    set_seed(config.seed)

    _, _, test_dataset = load_datasets_wiki_727(
        dataset_root=config.wiki_727_dir,
        model_id=config.model_id,
        max_seq_len=config.wiki_727_max_tokens,
        max_samples=config.max_samples,
        load_train=False,
        load_dev=False,
    )
    
    experiment_id = get_experiment_id(prefix="eval")
    output_dir = str(os.path.join(config.output_dir, experiment_id))

    inferrer_config = TocLlmInferrerConfig(checkpoint=config.checkpoint)
    inferrer = TocLlmInferrer(inferrer_config)
    test_results, test_metrics = infer_on_test_set(inferrer, test_dataset)
    if config.bootstrap_metrics:
        print(f"Bootstrapping {config.num_bootstrap_samples} times...", flush=True)
        bootstrapped_metrics = bootstrap_metrics(test_metrics, num_samples=config.num_bootstrap_samples, use_weighted_avg=False)
        aggregated_metrics = aggregate_metrics_with_stddev(bootstrapped_metrics)
    else:
        print(f"Aggregating metrics...", flush=True)
        aggregated_metrics = merge_metrics(test_metrics, use_weighted_avg=False)
    for m, val in aggregated_metrics.items():
        if isinstance(val, dict):
            print(f"- {m}: {round(val['avg']*100, 2)} +/- {round(val['std_dev']*100, 2)}")
        else:
            print(f"- {m}: {round(val*100, 2)}")
    os.makedirs(output_dir)
    print(f"Experiment ID: {experiment_id}", flush=True)
    results_filepath = os.path.join(output_dir, "results.json")
    with open(results_filepath, "w") as f:
        json.dump(test_results, f)
    print(f"Predictions and targets saved to: {results_filepath}", flush=True)
    metrics_filepath = os.path.join(output_dir, "metrics.json")
    with open(metrics_filepath, "w") as f:
        json.dump(aggregated_metrics, f, indent=4)
    print(f"Evaluation metrics saved to: {metrics_filepath}", flush=True)


def main():
    max_split_size = 256
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = f"max_split_size_mb:{max_split_size}"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-file", type=str, required=True,
        help="Path to the training config file (JSON). "
             "Must include the path to the wiki-727 dataset and the checkpoint path."
    )
    args = parser.parse_args()
    assert os.path.isfile(args.config_file), "Provided config file does not exist."
    print(f"Using settings from config file: {args.config_file}", flush=True)
    custom_config = load_json_data(args.config_file)
    config = EvalConfigWiki727(**custom_config)
    run_evaluation(config)


if __name__ == "__main__":
    main()

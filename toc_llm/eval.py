from __future__ import annotations
from collections import defaultdict
import warnings
import random

import numpy as np
from sklearn.metrics import precision_recall_fscore_support, matthews_corrcoef
from nltk.metrics import segmentation
import segeval


def get_half_segment_size(targets: list[int | float], start_labels: bool = False) -> int:
    num_segments = sum(t > 0 for t in targets) + 1
    if start_labels:
        num_elements = len(targets)
    else:
        num_elements = len(targets) + 1
    half_segment_size = 0.5 * (num_elements / num_segments)
    return max(1, round(half_segment_size))


class Metrics:
    """
    Class to compute and store segmentation metrics.
    It computes precision, recall, F1 score, Matthews correlation coefficient (MCC),
    Pk, GHD, and window difference (window_diff) metrics.
    Optionally, it can compute metrics using the segeval library.
    
    :param predictions: list of predicted change points (0/1 labels)
    :param targets: list of true change points (0/1 labels)
    :param window_width: width of the window for Pk and GHD metrics, if None, it is computed based on targets
    :param name: name of the metrics instance, useful for identification
    :param f1_average: type of averaging for F1 score, can be 'binary', 'micro', 'macro', or 'weighted'
    :param start_labels: whether the labels are start labels (True) or change labels (False).
        - Start labels mean that we have a label for every sentence, 
          and the label indicates whether the sentence is a start of a new topic.
        - Change labels mean that we have a label for every inter sentence boundary, 
          and the label indicates whether there is a change at that boundary.
    :param add_segeval_metrics: whether to compute additional metrics using the segeval library
    :return: an instance of Metrics class with computed metrics
    """

    def __init__(self, predictions: list[int], targets: list[int], window_width: int | None = None,
                 name: str | None = None, f1_average: str = "binary", start_labels: bool = False,
                 add_segeval_metrics: bool = False):
        self.predictions = predictions
        self.targets = targets
        self.name = name
        self.f1_average = f1_average
        self.start_labels = start_labels
        self.add_segeval_metrics = add_segeval_metrics
        self.precision, self.recall, self.f1, _ = precision_recall_fscore_support(
            targets, predictions, pos_label=1, average=f1_average, zero_division=0, labels=[0, 1])
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            self.mcc = matthews_corrcoef(targets, predictions)
        if window_width is None:
            window_width = get_half_segment_size(targets, start_labels)
        if window_width > len(targets):
            # No boundaries in targets, use one full window
            window_width = len(targets)
        self.pk = segmentation.pk(ref=targets, hyp=predictions, k=window_width, boundary=1)
        self.window_diff = segmentation.windowdiff(predictions, targets, k=window_width, boundary=1)
        mean_segment_length = len(targets) / (sum(targets) + 1)
        # print("Mean segment length:", mean_segment_length)
        self.ghd = segmentation.ghd(ref=targets, hyp=predictions, ins_cost=mean_segment_length,
                                    del_cost=mean_segment_length, shift_cost_coeff=2.0, boundary=1)
        # Compute metrics using segeval
        if self.add_segeval_metrics:
            segeval_metrics = compute_using_segeval(predictions, targets, start_labels)
            self.pk_segeval = segeval_metrics["Pk"]
            self.boundary_similarity = segeval_metrics["B"]
        self.hwd = None  # metric can be added after initialization

    def set_hwd(self, hwd: float):
        self.hwd = hwd

    def to_dict(self, include_sequences: bool = False) -> dict[str, float | None]:
        data = {f"f1_{self.f1_average}": self.f1, "precision": self.precision, "recall": self.recall, "mcc": self.mcc,
                "window_diff": self.window_diff, "pk": self.pk, "ghd": self.ghd, "name": self.name}
        if self.add_segeval_metrics:
            data.update({"pk_segeval": self.pk_segeval, "b": self.boundary_similarity})
        if self.hwd is not None:
            data["hwd"] = self.hwd
        if include_sequences:
            data.update({"predictions": self.predictions, "targets": self.targets})
        return data


def merge_metrics(metrics: list[Metrics], round_values: int | None = None, use_weighted_avg: bool = False
                  ) -> dict[str, float]:
    """
    Merge multiple Metrics instances into a single dictionary of aggregated metrics.
    
    :param metrics: list of Metrics instances to merge
    :param round_values: number of decimal places to round the metrics values, if None, no rounding is applied
    :param use_weighted_avg: whether to use weighted averages for the metrics.
        - If True, the metrics are weighted by the number of changes in each sample.
        - If False, all metrics are treated equally (unweighted average).
    :return: dict with aggregated metrics
    """
    if use_weighted_avg:
        # Normalize in relation to number of changes in video
        normalize_factor = 1 / sum([sum(m.targets) for m in metrics])
        metric_weights = [sum(m.targets) * normalize_factor for m in metrics]
    else:
        metric_weights = [1 / len(metrics) for _ in metrics]
    f1_average = metrics[0].f1_average
    if use_weighted_avg:
        # Weighted averages of classification metrics, use all targets and predictions
        all_targets, all_predictions = [], []
        for metric in metrics:
            all_targets.extend(metric.targets)
            all_predictions.extend(metric.predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_targets, all_predictions, pos_label=1, average=f1_average, zero_division=0, labels=[0, 1])
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            mcc = matthews_corrcoef(all_targets, all_predictions)
    else:
        # Compute unweighted averages (every video has equal weight)
        f1 = sum([m.f1 * metric_weights[n] for n, m in enumerate(metrics)])
        precision = sum([m.precision * metric_weights[n] for n, m in enumerate(metrics)])
        recall = sum([m.recall * metric_weights[n] for n, m in enumerate(metrics)])
        mcc = sum([m.mcc * metric_weights[n] for n, m in enumerate(metrics)])

    # Aggregate window metrics either equally (unweighted) or weighted by number of changes in video
    overall_metrics = {
        f"f1_{f1_average}": f1,
        "precision": precision,
        "recall": recall,
        "mcc": mcc,
        "window_diff": sum([m.window_diff * metric_weights[n] for n, m in enumerate(metrics)]),
        "pk": sum([m.pk * metric_weights[n] for n, m in enumerate(metrics)]),
        "ghd": sum([m.ghd * metric_weights[n] for n, m in enumerate(metrics)])
    }
    if metrics[0].add_segeval_metrics:
        m_update = {
            "pk_segeval": sum([m.pk_segeval * metric_weights[n] for n, m in enumerate(metrics)]),
            "b": sum([m.boundary_similarity * metric_weights[n] for n, m in enumerate(metrics)])
        }
        overall_metrics.update(m_update)
    if metrics[0].hwd is not None:
        overall_metrics["hwd"] = sum([m.hwd * metric_weights[n] for n, m in enumerate(metrics)])
    if round_values is not None:
        return {m: round(val, round_values) for m, val in overall_metrics.items()}
    return overall_metrics


def compute_using_segeval(predictions: list[int], targets: list[int], start_labels: bool) -> dict[str, float]:
    """
    Compute segmentation metrics using segeval library.

    :param predictions: list of predicted change points
    :param targets: list of true change points
    :param start_labels: whether the labels are start labels (True) or change labels (False)
    :return: dict with segmentation metrics
    """
    hyp_masses = topic_starts_to_masses(predictions, start_labels)
    ref_masses = topic_starts_to_masses(targets, start_labels)
    pk = segeval.pk(hyp_masses, ref_masses)
    try:
        bs = segeval.boundary_similarity(hyp_masses, ref_masses)
    except ValueError as e:
        bs = 0.0
        print(f"\nError in boundary similarity calculation, using 0.0 as fallback value: "
              f"{e}\nhyp_masses: {hyp_masses}\nref_masses: {ref_masses}\n", flush=True)
    return {"Pk": float(pk), "B": float(bs)}


def topic_starts_to_masses(label_seq, start_labels: bool) -> list[int]:
    """
    Convert a list of 0/1 topic-start/change labels into a list of segment lengths.
    This function returns the list of segment lengths suitable for segeval.
    """
    if start_labels:
        topic_starts = label_seq
    else:
        topic_starts = [0] + label_seq
    segment_lengths = []
    current_topic_start = 0  # We'll store the index where the current segment starts
    for i, is_topic_start in enumerate(topic_starts):
        if is_topic_start == 1:
            current_segment_length = i - current_topic_start
            segment_lengths.append(current_segment_length)
            # Start a new topic at i
            current_topic_start = i
    # Add final segment length
    last_segment_length = len(topic_starts) - current_topic_start
    segment_lengths.append(last_segment_length)
    return segment_lengths


def hierarchical_metric(hyp: dict[int, list[int]], ref: dict[int, list[int]]) -> float:
    """
    Computes a hierarchical F1 metric given:
      - hyp: { level_hyp -> [0/1 list of boundary tags] }
      - ref: { level_ref -> [0/1 list of boundary tags] }
    Each dictionary maps from a level (int) to a list of 0/1 boundary indicators for that level.

    Returns a single float: the maximum average F1 across all reference levels,
    where each reference level is matched to a hypothesis level that is the same
    or "finer" (i.e., we cannot move back up to coarser levels on the hypothesis side).

    The overall approach is:
      1) Sort ref levels -> LRef
      2) Sort hyp levels -> LHyp
      3) Build a 2D table f1_table[i][j] = F1(ref[LRef[i]], hyp[LHyp[j]])
      4) DP approach to choose a sequence of hypothesis levels h_1 ... h_{nRef}
         with h_{l+1} >= h_l, maximizing sum of f1_table[i][h_i].
      5) Average the maximum sum by dividing by nRef.
    """
    hyp = {lvl: seq[1:] for lvl, seq in hyp.items()}  # Remove the start label
    ref = {lvl: seq[1:] for lvl, seq in ref.items()}  # Remove the start label

    # 1) Sort the keys of ref and hyp
    levels_ref = sorted(ref.keys())
    levels_hyp = sorted(hyp.keys())

    nRef = len(levels_ref)
    nHyp = len(levels_hyp)

    if nRef == 0 or nHyp == 0:
        # If there is no level data in reference or hypothesis, return 0
        return 0.0

    # 2) Precompute F1 values for each (reference_level, hypothesis_level) pair
    #    We store this in f1_table[i][j], where i is index in levels_ref, and j is index in levels_hyp.
    f1_table = [[0.0 for _ in range(nHyp)] for _ in range(nRef)]
    for i, ref_lvl in enumerate(levels_ref):
        ref_seq = ref[ref_lvl]
        for j, hyp_lvl in enumerate(levels_hyp):
            hyp_seq = hyp[hyp_lvl]
            metric = Metrics(hyp_seq, ref_seq, start_labels=False, add_segeval_metrics=True)
            f1_table[i][j] = metric.boundary_similarity

    # 3) Define a DP table dp[i][j] = max sum of F1 from the first i+1 reference levels,
    #    where the i-th reference level is matched to the j-th hypothesis level.
    #    Then dp[i][j] must come from dp[i-1][k] for some k <= j.
    dp = [[0.0 for _ in range(nHyp)] for _ in range(nRef)]

    # 4) Base case: for i=0, dp[0][j] = f1_table[0][j] (matching level_ref[0] to level_hyp[j])
    for j in range(nHyp):
        dp[0][j] = f1_table[0][j]

    # 5) Fill the DP for i from 1..nRef-1
    for i in range(1, nRef):
        # We keep a running "prefix max" so we can do: dp[i][j] = max_{k in [0..j]} dp[i-1][k] + f1_table[i][j]
        prefix_max = dp[i-1][0]
        dp[i][0] = dp[i-1][0] + f1_table[i][0]
        for j in range(1, nHyp):
            prefix_max = max(prefix_max, dp[i-1][j])
            dp[i][j] = prefix_max + f1_table[i][j]

    # 6) The best alignment for all LRef levels is the max in dp[nRef-1][*]
    best_sum = max(dp[nRef-1])

    # 7) Average by the number of reference levels
    return best_sum / nRef


METRICS_OF_INTEREST = ["f1_binary", "precision", "recall", "pk", "pk_segeval", "b", "hwd"]


def aggregate_metrics_with_stddev(metrics: list[dict]) -> dict[str, dict[str, float]]:
    """
    Aggregate metrics from a list of dictionaries and return the average +/- standard deviation for each metric.
    
    :param metrics: list of dictionaries containing metrics
    :return: dict with aggregated metrics and their standard deviations
    """
    metric_2_values = defaultdict(list)
    for m in metrics:
        for k, v in m.items():
            if k in METRICS_OF_INTEREST:
                metric_2_values[k].append(v)
    aggregated_metrics = {}
    for m, vs in metric_2_values.items():
        avg = np.average(vs)
        std_dev = np.std(vs)
        aggregated_metrics[m] = {"avg": avg, "std_dev": std_dev}
    return aggregated_metrics


def bootstrap_metrics(metrics: list[Metrics], num_samples: int = 100, 
                      round_values: int | None = None, use_weighted_avg: bool = False) -> list[dict[str, float]]:
    """
    Bootstrap the metrics by generating random samples from the provided metrics.

    :param metrics: list of Metrics instances to bootstrap
    :param num_samples: number of bootstrap samples to generate
    :param round_values: number of decimal places to round the metrics values, if None, no rounding is applied
    :param use_weighted_avg: whether to use weighted averages for the metrics.
        - If True, the metrics are weighted by the number of changes in each sample.
        - If False, all metrics are treated equally (unweighted average).
    :return: list of dictionaries with bootstrapped metrics
    """
    bootstrapped_metrics = []
    num_metrics = len(metrics)
    for n in range(num_samples):
        random.seed(n)  # Set seed to always use the same bootstrap at iteration n
        bootstrapped_sample = random.choices(metrics, k=num_metrics)
        avg_metrics = merge_metrics(bootstrapped_sample, round_values=round_values, use_weighted_avg=use_weighted_avg)
        bootstrapped_metrics.append(avg_metrics)
    return bootstrapped_metrics

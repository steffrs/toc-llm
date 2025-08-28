from __future__ import annotations
import json
from typing import List, Optional, Dict

import torch
from pydantic import BaseModel, conint, conlist

from ..prompts import SYSTEM_PROMPT, SYSTEM_PROMPT_PAUSE, USER_PROMPT, FEW_SHOT_EXTENSION


def load_datasets() -> tuple[TocDataset, TocDataset, TocDataset]:
    raise NotImplementedError("Implement this function for your specific datasets. "
                              "Look at the `TocDataset` class below for the expected format.")
    print("Loading TRAIN dataset...", flush=True)
    train_dataset = TocDataset()
    print("Loading DEV dataset...", flush=True)
    dev_dataset = TocDataset()
    print("Loading TEST dataset...", flush=True)
    test_dataset = TocDataset()
    print(f"\nNumber of samples:\n"
          f" - Train: {len(train_dataset)}\n"
          f" - Dev: {len(dev_dataset)}\n"
          f" - Test: {len(test_dataset)}\n", flush=True)
    return train_dataset, dev_dataset, test_dataset


class Topic(BaseModel):
    title: str
    start: conint(ge=0)
    subtopics: Optional[List["Topic"]] = None

    class Config:
        from_attributes = True


class TableOfContents(BaseModel):
    topics: conlist(Topic, min_length=1)

    class Config:
        from_attributes = True


def serialize_toc(toc_dict: dict) -> str:
    """Convert nested toc_dict (with "topics", "title", "start", "subtopics") into a JSON string."""
    return json.dumps(toc_dict, ensure_ascii=False)


def sentences_to_llm_format(sentences: list[str], pauses: list[float] | None = None) -> str:
    lines = []
    add_pause = pauses is not None
    for n, sentence in enumerate(sentences):
        if add_pause:
            pause = round(pauses[n], 2)
            if pause > 0.0:
                line = f"{n} (pause={pause}s): {sentence}"
            else:
                line = f"{n}: {sentence}"
        else:
            line = f"{n}: {sentence}"
        lines.append(line)
    return "\n".join(lines)


def create_messages(transcript: str, toc: dict | str | None, is_mistral: bool = False,
                    few_shot_samples: list[tuple[str, str]] | None = None, add_pause: bool = False
                    ) -> list[dict]:
    if add_pause:
        system_prompt = SYSTEM_PROMPT_PAUSE
    else:
        system_prompt = SYSTEM_PROMPT
    user_prompt = USER_PROMPT
    user_prompt = user_prompt.replace("<TRANSCRIPT>", transcript)
    if few_shot_samples is not None:
        few_shot_extension = FEW_SHOT_EXTENSION
        for n, (example_transcript, example_toc) in enumerate(few_shot_samples):
            few_shot_extension = few_shot_extension.replace(f"<TRANSCRIPT_EXAMPLE_{n + 1}>", example_transcript)
            few_shot_extension = few_shot_extension.replace(f"<TOC_EXAMPLE_{n + 1}>", example_toc)
        system_prompt += f"\n{few_shot_extension}"
    if is_mistral:
        messages = [
            {"role": "user", "content": f"{system_prompt}\n\n{user_prompt}"},  # Reduced Mistral format
        ]
    else:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    if toc is not None:
        if isinstance(toc, dict):
            toc = serialize_toc(toc)
        messages.append({"role": "assistant", "content": toc})
    return messages


def messages_to_tokenized(messages: list[dict] | list[list[dict]], tokenizer, max_length, return_assistant_tm: bool):
    """Tokenize messages or batch of messages."""
    add_generation_prompt = messages[-1]["role"] != "assistant"
    tokenized = tokenizer.apply_chat_template(
        conversation=messages,
        tokenize=True,
        truncation=True,
        max_length=max_length,
        padding="longest",
        return_tensors="pt",
        add_generation_prompt=add_generation_prompt,
        return_dict=True,
        return_assistant_tokens_mask=return_assistant_tm,
    )
    return tokenized


def compute_pause_durations(intervals: list[tuple[float, float]]) -> list[float]:
    """
    Compute the pause durations between consecutive sentence intervals.
    Returns a list of pause durations, where the first element is always 0.0 (pause before the first sentence).
    """
    pauses = [0.0]  # Start with a zero pause before the first sentence
    for i in range(len(intervals) - 1):
        pause_duration = intervals[i + 1][0] - intervals[i][1]
        pauses.append(pause_duration)
    return pauses


class TocDataset(torch.utils.data.Dataset):
    """
    Torch Dataset to load toc data.
    Expects:
    - transcripts: List of lists of sentences (each sentence is a string).
    - tocs: List of table of contents (str), of the following format:
            1. Introduction [0]
            1.1 Project Goals [5]
            1.2 Background [10]
            2. Another Main Topic [15]
            2.1 Deep Dive [20]
            2.2 Summary [25]
    - sentence_intervals: Optional list of lists of tuples (start_seconds, end_seconds) for each sentence.
    """

    def __init__(self, transcripts: list[list[str]], tocs: list[str],
                 sentence_intervals: list[list[tuple[float, float]]] | None = None):
        self._samples = []
        add_pause = sentence_intervals is not None
        for n, sent_seq in enumerate(transcripts):
            if len(sent_seq) <= 2:
                print(f"Skipping sample, since it contains <=2 sentences.", flush=True)
                continue
            if add_pause:
                itv_seq = sentence_intervals[n]
                pauses = compute_pause_durations(itv_seq)
            else:
                pauses = None
            transcript = sentences_to_llm_format(sent_seq, pauses=pauses)
            toc = tocs[n]
            self._samples.append((transcript, toc))

    def __getitem__(self, idx) -> tuple[str, str]:
        return self._samples[idx]

    def __len__(self) -> int:
        return len(self._samples)


def _traverse_topics(
        topic: Topic,
        depth: int,
        label_dict: Dict[int, List[int]],
        total_sentences: int
) -> None:
    """
    Recursively visit each topic, marking topic.start = 1 in label_dict for the given depth. 
    Also visits subtopics at depth+1.
    """
    if depth not in label_dict:
        label_dict[depth] = [0] * total_sentences
    if topic.start < total_sentences:  # if it is >= -> out of index error
        label_dict[depth][topic.start] = 1
    # Recurse into subtopics
    if topic.subtopics:
        for sub in topic.subtopics:
            _traverse_topics(sub, depth + 1, label_dict, total_sentences)


def build_topic_start_labels(
        toc: TableOfContents,
        total_sentences: int,
) -> Dict[int, List[int]]:
    """
    Build a dictionary that maps each hierarchical level to a list of binary
    topic-change labels (0 or 1).

    Example output:
    {
      1: [0, 0, 1, 0, 0],
      2: [0, 0, 1, 0, 1],
      3: [ ... ]
    }

    Steps:
    1) Recursively traverse all topics, marking the start index for each depth.
    2) After traversal, propagate any 1's from level L to levels L+1..max_depth.
    """
    # Dictionary: level -> list of binary labels
    label_dict: Dict[int, List[int]] = {}

    top_level_depth = 2 if len(toc.topics) > 1 else 1
    for top_topic in toc.topics:
        _traverse_topics(top_topic, top_level_depth, label_dict, total_sentences)

    # Propagate any topic changes from shallow to deeper levels.
    if not label_dict:
        print("No labels found in Table of Contents.", flush=True)
        return label_dict
    max_depth = max(label_dict.keys())
    for depth in range(top_level_depth, max_depth):
        if depth not in label_dict:
            continue  # check for security
        for idx, val in enumerate(label_dict[depth]):
            if val == 1:
                # Set deeper levels to 1 at this sentence index
                for deeper in range(depth + 1, max_depth + 1):
                    if deeper in label_dict:
                        label_dict[deeper][idx] = 1

    return label_dict


def get_test_tocs() -> tuple[TableOfContents, TableOfContents]:
    example_toc = TableOfContents(
        topics=[
            Topic(
                title="Chapter 1",
                start=2,
                subtopics=[
                    Topic(title="Section 1.1", start=4),
                    Topic(
                        title="Section 1.2",
                        start=6,
                        subtopics=[
                            Topic(title="Subsection 1.2.a", start=7)
                        ]
                    )
                ]
            ),
            Topic(
                title="Chapter 2",
                start=9
            )
        ]
    )
    example_toc_2 = TableOfContents(
        topics=[
            Topic(
                title="Chapter 1",
                start=2,
                subtopics=[
                    Topic(title="Section 1.1", start=2),
                    Topic(title="Section 1.2", start=8),
                ]
            ),
            Topic(
                title="Chapter 2",
                start=9,
                subtopics=[Topic(title="Section 2.1", start=10)]
            )
        ]
    )
    return example_toc, example_toc_2


def toc_to_label_dict(toc_text: str, total_sentences: int) -> Dict[int, List[int]]:
    """
    Given a textual ToC, e.g.:
        1. Introduction [0]
        1.1 Project Goals [5]
        1.2 Background [10]
        2. Another Main Topic [15]
        2.1 Deep Dive [20]
        2.2 Summary [25]

    Returns a dictionary: level -> [0/1] * total_sentences
    Where each list has a '1' at indices where a new topic at that level starts.

    Special Logic:
      - The 'level' is determined by how many dotted segments there are. e.g. "1.2.3" => level=3
      - If the ToC jumps from e.g. level=1 to level=3, we fill in the missing level=2
        boundary at the same start index (and also inherit to deeper levels).
      - A new topic at level L also sets a boundary at levels L+1..max_level (inheritance).
        (You can comment out that portion if you do NOT want deeper levels to automatically reset.)
    """
    parsed = []
    lines = toc_text.strip().split("\n")
    lines = [line.strip() for line in lines if line.strip()]
    for line in lines:
        line_split = line.split()
        numbered = line_split[0]
        numbered_parts = numbered.split(".")
        # every part must have at least one digit character, otherwise it is not a level descriptor
        numbered_parts = [n for n in numbered_parts if any(c.isdigit() for c in n)]
        level = len(numbered_parts)
        if level == 0:
            print(f"Skipping line '{line}' due to invalid level format.", flush=True)
            continue
        sent_ref = line_split[-1].strip()
        try:
            start_idx = int(sent_ref.strip("[]"))
        except ValueError:
            print(f"Skipping line '{line}' due to invalid start index format.", flush=True)
            continue
        parsed.append((level, start_idx))

    if not parsed:
        return {}  # No valid lines found -> return empty dictionary

    # Determine the max level so we know how many label lists to create
    max_level = max(x[0] for x in parsed)

    # Initialize the label_dict with zeros
    label_dict = {lvl: [0]*total_sentences for lvl in range(1, max_level+1)}

    # Sort entries by start_idx to process in chronological order
    parsed.sort(key=lambda x: x[1])

    # Fill the label dict with skip-level logic + deeper-level inheritance
    prev_level = None
    for (lvl, start_idx) in parsed:
        if start_idx < total_sentences:
            if prev_level is not None and lvl > prev_level + 1:
                # We jumped from prev_level -> lvl, fill in missing levels
                for missing_lvl in range(prev_level+1, lvl+1):
                    # Mark the new boundary at missing_lvl
                    label_dict[missing_lvl][start_idx] = 1
                    # Inherit down deeper levels
                    for deeper_lvl in range(missing_lvl+1, max_level+1):
                        label_dict[deeper_lvl][start_idx] = 1
            else:
                # Normal or backward jump -> just set boundary at lvl
                label_dict[lvl][start_idx] = 1
                # Inherit down deeper levels
                for deeper_lvl in range(lvl+1, max_level+1):
                    label_dict[deeper_lvl][start_idx] = 1

            prev_level = lvl
        else:
            # If the start_idx is >= total_sentences, ignore it
            pass

    return label_dict


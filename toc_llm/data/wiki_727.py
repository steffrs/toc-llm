from __future__ import annotations
import os
from typing import Generator
from dataclasses import dataclass
from tqdm import tqdm

import torch
from transformers import AutoTokenizer


def load_datasets(dataset_root: str, model_id: str, max_seq_len: int | None, max_samples: int | None,
                  load_train: bool = True, load_dev: bool = True, load_test: bool = True,
                  ) -> tuple[Wiki727Dataset, Wiki727Dataset, Wiki727Dataset]:
    if max_seq_len is not None:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
    else:
        tokenizer = None
    if load_train:
        print("Loading TRAIN dataset...", flush=True)
        train_dir = os.path.join(dataset_root, "train")
        dataset_train = Wiki727Dataset(train_dir, max_seq_len, tokenizer, max_samples)
    else:
        dataset_train = None
    if load_dev:
        print("Loading DEV dataset...", flush=True)
        dev_dir = os.path.join(dataset_root, "dev")
        dataset_dev = Wiki727Dataset(dev_dir, max_seq_len, tokenizer, max_samples)
    else:
        dataset_dev = None
    if load_test:        
        print("Loading TEST dataset...", flush=True)
        test_dir = os.path.join(dataset_root, "test")
        dataset_test = Wiki727Dataset(test_dir, max_seq_len, tokenizer, max_samples)
    else:
        dataset_test = None
    print(f"\nNumber of samples:\n"
          f"- Train: {len(dataset_train) if dataset_train is not None else '-'}\n"
          f"- Dev: {len(dataset_dev) if dataset_dev is not None else '-'}\n"
          f"- Test: {len(dataset_test) if dataset_test is not None else '-'}\n",
          flush=True)
    return dataset_train, dataset_dev, dataset_test


def get_filepaths_recursively(root_path: str) -> Generator[str, None, None]:
    for name in os.listdir(root_path):
        path = os.path.join(root_path, name)
        if os.path.isdir(path):
            for fp in get_filepaths_recursively(path):
                yield fp
        elif os.path.isfile(path):
            yield path


@dataclass
class Sentence:
    text: str  # sentence in the Wikipedia article
    topic: str  # topic name (title)
    level: int  # hierarchical level, starting from 1 (most upper level, usually the intro section in Wikipedia article)
    is_start: bool


def new_topic_dict(sentence: Sentence, sentence_index: int) -> dict:
    """
    Create a basic dictionary for a new topic, given the Sentence
    that starts it, along with the sentence index in the article.
    """
    return {
        "title": sentence.topic,
        "start": sentence_index,
        "subtopics": []
    }


def build_nested_topics(sentences: list[Sentence]) -> list[dict]:
    """
    Build a hierarchical list of topic dictionaries from the given sentences.
    Each topic dict has the form:
      {
        "title": <str>,
        "start": <int>,
        "subtopics": [more dicts...]   <-- only present if subtopics exist
      }
    Returns a list of top-level topic dicts.
    """
    top_level_topics: list[dict] = []
    stack: list[dict] = []

    for i, s in enumerate(sentences):
        if s.is_start or i == 0:
            # Create a new topic dict without a "subtopics" key
            # We'll only add "subtopics" if we attach a child to it.
            new_topic = {
                "title": s.topic,
                "start": i
            }

            # If the new topic is level 2, it belongs at the top level.
            # Otherwise, pop from the stack until we have a suitable parent.
            while len(stack) >= s.level -1:
                stack.pop()

            if not stack:
                # No parent on the stack => top-level topic
                top_level_topics.append(new_topic)
            else:
                # Attach to the "subtopics" of the current top of the stack.
                parent = stack[-1]
                parent.setdefault("subtopics", []).append(new_topic)

            # Push the new topic onto the stack
            stack.append(new_topic)

    return top_level_topics


def build_toc_dict(sentences: list[Sentence]) -> dict:
    """
    Build the final dictionary structure.
    """
    top_level_topics = build_nested_topics(sentences)
    return {
        "topics": top_level_topics
    }


@dataclass
class Header:
    title: str
    level: int
    start: int | None = None

@dataclass
class Sent:
    idx: int
    text: str


def _load_from_lines(lines: list[str]) -> tuple[str, str] | tuple[None, None]:
    # 1. Remove preface if it exists
    #    If the first line contains 'preface', remove lines until we hit a heading line ("=====").
    if "preface" in lines[0].lower():
        lines = lines[1:]
        while lines and not lines[0].startswith("====="):
            lines = lines[1:]
        if not lines:
            # If we removed everything, return None
            return None, None

    all_items = []
    sent_idx = 0
    # 2. Parse each line
    #    - If it's a heading (starts with "======"), create a Header.
    #    - Otherwise, it's a sentence; create a Sent and increment sent_idx.
    for line in lines:
        if line.startswith("======"):
            line_split = line.split(",", maxsplit=2)
            if len(line_split) < 3:
                # bad format
                return None, None

            level_str = line_split[1].strip()
            if not level_str.isdigit():
                return None, None

            current_level = int(level_str)
            topic = line_split[2].strip(".").strip()

            # Subtract 1 so that original level=2 => new level=1
            # (ensuring "preface" headers at level=1 are effectively removed).
            header = Header(title=topic, level=current_level - 1)
            all_items.append(header)
        else:
            # Normal sentence line
            sent = Sent(idx=sent_idx, text=line)
            sent_idx += 1
            all_items.append(sent)

    # 3. Determine each header's 'start' = first sentence index after it
    headers = []
    for i, itm in enumerate(all_items[:-1]):
        if isinstance(itm, Header):
            for nxt_itm in all_items[i+1:]:
                if isinstance(nxt_itm, Sent):
                    itm.start = nxt_itm.idx
                    headers.append(itm)
                    break

    # 4. Build the dotted-number ToC
    #    Sort headers by their 'start' index (chronological).
    headers = [h for h in headers if h.level > 0 and h.start is not None]
    headers.sort(key=lambda h: h.start)

    numbering_counters = [0] * 20
    toc_lines = []

    for hdr in headers:
        # If a header has level < 1, skip it (or treat it as level=1).
        # This can happen if the original was level=1 => became 0. We skip those.
        if hdr.level < 1:
            continue

        numbering_counters[hdr.level - 1] += 1
        # Reset deeper levels
        for deeper_l in range(hdr.level, len(numbering_counters)):
            numbering_counters[deeper_l] = 0

        # Build dotted prefix, e.g. "1.2.1"
        dotted_parts = []
        for l_idx in range(hdr.level):
            if numbering_counters[l_idx] > 0:
                dotted_parts.append(str(numbering_counters[l_idx]))
        dotted_str = ".".join(dotted_parts)

        toc_line = f"{dotted_str} {hdr.title} [{hdr.start}]"
        toc_lines.append(toc_line)

    toc_str = "\n".join(toc_lines)

    # 5. Build the final transcript (only the Sent items), line = "{idx}: {text}"
    sents = [x for x in all_items if isinstance(x, Sent)]
    transcript_lines = [f"{s.idx}: {s.text}" for s in sents]
    transcript_str = "\n".join(transcript_lines)

    return transcript_str, toc_str


def load_from_filepath(filepath: str) -> tuple[str, dict] | tuple[None, None]:
    try:
        with open(filepath, encoding="utf-8") as inp:
            lines = [line.strip() for line in inp.readlines() if line.strip()]
    except Exception as e:
        print(f"Error loading file '{filepath}': {e}", flush=True)
        return None, None
    return _load_from_lines(lines)


def load_from_text(text: str) -> tuple[str, dict] | tuple[None, None]:
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    return _load_from_lines(lines)


def add_initial_subtopics_if_missing(topics: list[dict]) -> list[dict]:

    def add_initial_subtopics_recursively(topics_: list[dict]) -> None:
        for topic_ in topics_:
            if "subtopics" in topic_:
                if topic_["subtopics"][0]["start"] != topic_["start"]:
                    # need to add initial subtopic here that has same start as super topic
                    initial_subtopic = {"title": "Introduction", "start": topic_["start"]}
                    topic_["subtopics"].insert(0, initial_subtopic)
                add_initial_subtopics_recursively(topic_["subtopics"])

    add_initial_subtopics_recursively(topics)
    return topics


class Wiki727Dataset(torch.utils.data.Dataset):
    """
    Torch Dataset to load topic-annotated Wiki727 data.
    """

    def __init__(self, data_path: str, max_seq_len: int | None = None,
                 tokenizer=None, max_samples: int | None = None):
        if max_seq_len is not None and tokenizer is not None:
            filter_for_max_seq_len = True
            print(f"- Filter out text samples with more than {max_seq_len} tokens...", flush=True)
            num_filtered_out = 0
        else:
            filter_for_max_seq_len = False
        self._samples = []
        filepaths = list(get_filepaths_recursively(data_path))
        filepaths.sort()
        num_samples = 0
        for filepath in tqdm(filepaths, desc="Loading Wiki727 samples"):
            text, toc = load_from_filepath(filepath)
            if text is None:
                continue
            if filter_for_max_seq_len:
                output = tokenizer(
                    text,
                    add_special_tokens=False,
                    return_length=True,
                    truncation=False
                )
                num_tokens = output["length"][0]
                if num_tokens > max_seq_len:
                    num_filtered_out += 1
                    continue
            self._samples.append((text, toc))
            num_samples += 1
            if max_samples is not None and num_samples == max_samples:
                break
        if filter_for_max_seq_len:
            print(f"- Num filtered out due to max_seq_len: {num_filtered_out}", flush=True)

    def __getitem__(self, idx):
        return self._samples[idx]

    def __len__(self):
        return len(self._samples)

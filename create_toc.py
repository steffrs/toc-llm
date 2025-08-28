import argparse
import os

from toc_llm.utils import set_seed
from toc_llm.inferrer import TocLlmInferrer, TocLlmInferrerConfig


def generate_toc(input_file: str, checkpoint_dir: str, output_file: str | None = None):
    set_seed(0)
    config = TocLlmInferrerConfig(checkpoint=checkpoint_dir)
    inferrer = TocLlmInferrer(config)
    
    with open(input_file, "r") as f:
        sentences = f.readlines()
    sentences = [s.strip() for s in sentences if s.strip()]
    if len(sentences) <= 3:
        print(f"Input appears to be quite short (only {len(sentences)} sentences). "
              f"Please check the format of the input file, make sure to use one sentence per line.")
        return
    toc, _ = inferrer(sentences)
    if output_file is None:
        print("Generated table of contents:", flush=True)
        print("-" * 80, flush=True)
        print(toc, flush=True)
        print("-" * 80, flush=True)
        return
    with open(output_file, "w") as f:
        f.write(toc+"\n")
    print(f"Table of contents saved to: {output_file}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input-file", type=str, required=True,
        help="Input text file to create table-of-contents for. Expected format: one sentence per line."
    )
    parser.add_argument("-c", "--checkpoint", type=str,
                        default="./models/pre-trained_on_wiki-727", help="Directory containing the model checkpoint.")
    parser.add_argument("-o", "--output-file", type=str, required=False, default=None,
                        help="Output text file to save the generated table-of-contents. "
                             "If not provided, the table-of-contents will be printed to stdout.")
    args = parser.parse_args()
    assert os.path.isdir(args.checkpoint), f"Provided `--checkpoint`/`-c` '{args.checkpoint_dir}' does not exist."
    assert os.path.isfile(args.input_file), f"Provided `--input-file`/`-i` '{args.input}' does not exist."
    if args.output_file is not None:
        assert not os.path.exists(args.output_file), (f"Provided `--output-file`/`-o` '{args.output_file}' already "
                                                      f"exists. Please delete it or provide a non-existing filepath.")
    generate_toc(args.input_file, args.checkpoint, args.output_file)


if __name__ == "__main__":
    main()

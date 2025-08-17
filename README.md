# toc-llm

Code repository for the paper
"Towards Multi-Level Transcript Segmentation: LoRA Fine-Tuning for Table-of-Contents Generation"
(Interspeech 2025).


## Inference
To create a table-of-contents for a text file, use the following command in python:
```bash
python create_toc.py -i /path/to/file.txt
```
You can provide the following arguments:
- `-i` or `--input-file`: Path to the input text file. Required format: One sentence/utterance per line.
- `-c` or `--checkpoint`: Path to the pre-trained LoRA directory to use for inference. 
  Default is `models/pre-trained_on_wiki-727`.
- `-o` or `--output-file` (Optional): Path to the output file where the table-of-contents will be written to. 
  If not provided, the output will be printed to the console. Default is `None`.

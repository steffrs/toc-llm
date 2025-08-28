# toc-llm

Code repository for the paper
"Towards Multi-Level Transcript Segmentation: LoRA Fine-Tuning for Table-of-Contents Generation"
(Interspeech 2025).

## Setup
Create a new environment (python=3.12) and install the dependencies:
```commandline
python -m pip install -r requirements.txt
```

## Inference
To create a table-of-contents for a text file, use the following command in python:
```commandline
python create_toc.py -i /path/to/file.txt
```
You can provide the following arguments:
- `-i` or `--input-file`: Path to the input text file. Required format: One sentence/utterance per line.
- `-c` or `--checkpoint`: Path to the pre-trained LoRA directory to use for inference. 
  Default is `models/pre-trained_on_wiki-727`.
- `-o` or `--output-file` (Optional): Path to the output file where the table-of-contents will be written to. 
  If not provided, the output will be printed to the console. Default is `None`.

## Citation
If you find this code useful, please cite our paper:
```bibtex
@inproceedings{freisinger25_interspeech,
  title     = {{Towards Multi-Level Transcript Segmentation: LoRA Fine-Tuning for Table-of-Contents Generation}},
  author    = {{Steffen Freisinger and Philipp Seeberger and Thomas Ranzenberger and Tobias Bocklet and Korbinian Riedhammer}},
  year      = {{2025}},
  booktitle = {{Interspeech 2025}},
  pages     = {{276--280}},
  doi       = {{10.21437/Interspeech.2025-2792}},
  issn      = {{2958-1796}},
}
```

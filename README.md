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
To create a table-of-contents for a text file, use the python script `create_toc.py`:
```commandline
python create_toc.py -i /path/to/file.txt
```
You can provide the following arguments:
- `-i` or `--input-file`: Path to the input text file. Required format: One sentence/utterance per line.
- `-c` or `--checkpoint`: Path to the pre-trained LoRA directory to use for inference. 
  Default is [`models/pre-trained_on_wiki-727k`](`models/pre-trained_on_wiki-727k`), 
  which includes model adapters for the LLM [`unsloth/Mistral-Nemo-Instruct-2407-bnb-4bit`](https://huggingface.co/unsloth/Mistral-Nemo-Instruct-2407-bnb-4bit),
  trained on the full Wiki-727k dataset for one epoch. 
- `-o` or `--output-file` (Optional): Path to the output file where the table-of-contents will be written to. 
  If not provided, the output will be printed to the console. Default is `None`.


## Training
To train a model on the Wiki-727k dataset, use the python script `train_on_wiki-727k.py`:
```commandline
python train_on_wiki-727k.py --config-file train_config.json
```
The `--config-file` argument specifies the path to a JSON configuration file that contains the training parameters.
The config file must contain the path to the Wiki-727k dataset (`wiki_727k_dir`).
Use the template [`train_config_template.json`](./train_config_template.json) as a reference.

If you want to train on your own dataset, you need to implement a custom `TocDataset` class. 
See [`toc_llm/data/toc.py`](./toc_llm/data/toc.py) for reference.
Also, adapt the python script `train_on_custom_dataset.py` according to your needs.


## Evaluation
To evaluate a model on the Wiki-727k dataset, use the python script `eval_on_wiki-727k.py`:
```commandline
python eval_on_wiki-727k.py --config-file eval_config.json
```
The `--config-file` argument specifies the path to a JSON configuration file that contains the evaluation parameters.
The config file must contain the path to the Wiki-727k dataset (`wiki_727k_dir`) 
and the path to the model checkpoint (`checkpoint`).
You can use the model checkpoint pre-trained on Wiki-727k in this repository: 
[`models/pre-trained_on_wiki-727k`](`models/pre-trained_on_wiki-727k`).
Use the template [`train_config_template.json`](./train_config_template.json) as a reference.

If you want to evaluate on your own dataset, you need to implement a custom `TocDataset` class. 
See [`toc_llm/data/toc.py`](./toc_llm/data/toc.py) for reference.
Also, adapt the python script `eval_on_custom_dataset.py` according to your needs.


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

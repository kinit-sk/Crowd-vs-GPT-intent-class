
# ChatGPT to Replace Crowdsourcing of Paraphrases for Intent Classification: Higher Diversity and Comparable Model Robustness

This is data for the paper: ["ChatGPT to Replace Crowdsourcing of Paraphrases for Intent Classification: Higher Diversity and Comparable Model Robustness"](https://arxiv.org/abs/2305.12947) which was accepted to the main track of [EMNLP'23](https://2023.emnlp.org/).

## Abstract

The emergence of generative large language models (LLMs) raises the question: what will be its impact on crowdsourcing. Traditionally, crowdsourcing has been used for acquiring solutions to a wide variety of human-intelligence tasks, including ones involving text generation, manipulation or evaluation. For some of these tasks, models like ChatGPT can potentially substitute human workers. In this study, we investigate, whether this is the case for the task of paraphrase generation for intent classification. We quasi-replicated the data collection methodology of an [existing crowdsourcing study](https://aclanthology.org/2020.emnlp-main.650/) (similar scale, prompts and seed data) using ChatGPT. We show that ChatGPT-created paraphrases are more diverse and lead to more robust models.


# Files

**Important note:** We provide a ``requirements.txt`` file for our setup and pytorch on CPU(!): please note that if you wish to run these experiments on your GPU to have the pytorch version conforming to your version of CUDA installed on your machine - details can be found [here](https://pytorch.org/get-started/locally/). For the syntax tree parsers please visit [CoreNLP site](https://stanfordnlp.github.io/CoreNLP/). We downloaded the .jar file and put it into the ``stanford_nlp/`` directory in our case.

Note: The data from the original study (found [here](https://aclanthology.org/2020.emnlp-main.650/)) should be downloaded and unpacked into the root. The json configuration files are used for the ChatGPT data collection and data preprocessing - having those data is required if you wish to run those steps.

### Diversity experiments

The directories:

``diversity_experiments/`` - contains all of the data and analysis in jupyter notebook form for the lexical and syntactical diversity comparison between the original study and ChatGPT.

``diversity_experiments/collected_responses_in_pkl/`` - contains the collected API responses from ChatGPT for each of the data collection round. Also contains data collected via Falcon-40b-instruct.

``diversity_experiments/datasets/`` - contains the parsed CSV files for the datasets used in the study.

The jupyter notebooks:

`diversity_experiments/collect_data_and_preprocess_chatgpt_and_orig_study.ipynb` - contains the code to make request to ChatGPT API and its parsing into csv files together with preprocessing of files from the original study.

`diversity_experiments/analysis of collected data.ipynb` - code for visualization and computation of both lexical and syntactical diversity.

Scripts:

`diversity_experiments/collect_falcon_data.py` - python script to collect data for Falcon-40b-instruct via quantization and PEFT methods.

### Model robustness on OOD experiments

The directories:

``ood_robustness_experiments/`` - contains all of the data and model training code in jupyter notebook format to collect data from ChatGPT, preprocess it and train SVM and BERT models on it for 5 different datasets.

``ood_robustness_experiments/challenge_data/`` - contains the parsed CSV files for the *human*, *original* and *gpt* data for all 5 datasets (fb, liu, snips, atis and clinc150).

The jupyter notebooks:

`diversity_experiments/scrap data *dataset_name* and train SVM.ipynb` - contains the code to make request to ChatGPT API and its parsing into csv files together with preprocessing of files from the original study for that given dataset. Also contains code for training SVM model.

`diversity_experiments/train bert *dataset_name*.ipynb` - code for training BERT large on the collected data. We recommend using a GPU for training with atleast 8GB of VRAM. We ran our experiments on a RTX3090 machine with 24 GB of RAM and 8 CPUs. Some of our experiments were run also on Kaggle with the GPU P100.

### Paper citing

If you wish to reference the paper, please wait for the published version which will be added in the next 2 months. Meanwhile, you can use this for your preprints:

```
@misc{cegin2023chatgpt,
      title={ChatGPT to Replace Crowdsourcing of Paraphrases for Intent Classification: Higher Diversity and Comparable Model Robustness}, 
      author={Jan Cegin and Jakub Simko and Peter Brusilovsky},
      year={2023},
      eprint={2305.12947},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
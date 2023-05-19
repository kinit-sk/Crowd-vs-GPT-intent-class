# ChatGPT to Replace Crowdsourcing of Paraphrases for Intent Classification: Higher Diversity and Comparable Model Robustness

This is repository for the paper: "ChatGPT to Replace Crowdsourcing of Paraphrases for Intent Classification: Higher Diversity and Comparable Model Robustness"

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

## Abstract

The emergence of generative large language models (LLMs) raises the question: what will be its impact on crowdsourcing. Traditionally, crowdsourcing has been used for acquiring solutions to a wide variety of human-intelligence tasks, including ones involving text generation, manipulation or evaluation. For some of these tasks, models like ChatGPT can potentially substitute human workers. In this study, we investigate, whether this is the case for the task of paraphrase generation for intent classification. We quasi-replicated the data collection methodology of an [existing crowdsourcing study](https://aclanthology.org/2020.emnlp-main.650/) (similar scale, prompts and seed data) using ChatGPT. We show that ChatGPT-created paraphrases are more diverse and lead to more robust models.


# Files

The directories contain:

``collected_responses_in_pkl/`` - contains the collected API responses from ChatGPT for each of the data collection round.

``gpt_datasets/`` - contains the parsed CSV files for the datasets used in the study. We only include the ChatGPT collected data without the data from the original study that can be found [here](https://aclanthology.org/2020.emnlp-main.650/). The main directory contains files used for data analysis and the ``for_train_test`` subdirectory contains splits used in training of models.

The jupyter notebooks

`collect_data_and_preprocess_chatgpt_and_orig_study.ipynb` - contains the code to make request to ChatGPT API and its parsing into csv files

`analysis of collected data.ipynb` - code for visualization and computation of both lexical and syntactical diversity

`train svm.ipynb` - code for training the SVM with tf-idf features

`train bert.ipynb` - code for training BERT large with the datasets

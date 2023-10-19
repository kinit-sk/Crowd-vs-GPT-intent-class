import json
import os
from pprint import pprint
import bitsandbytes as bnb
import torch
import torch.nn as nn
import transformers
import pickle
from datasets import load_dataset
from huggingface_hub import notebook_login
from peft import (
    LoraConfig,
    PeftConfig,
    PeftModel,
    get_peft_model,
    prepare_model_for_kbit_training
)
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

MODEL_NAME = "tiiuae/falcon-40b-instruct"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    trust_remote_code=True,
    load_in_8bit=True,
    torch_dtype=torch.bfloat16,
    quantization_config=bnb_config
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

generation_config = model.generation_config
generation_config.temperature = 0.9
generation_config.top_p = 0.8
generation_config.pad_token_id = tokenizer.eos_token_id
generation_config.eos_token_id = tokenizer.eos_token_id
generation_config.num_return_sequences = 13
generation_config.max_new_tokens = 250
generation_config.do_sample = True

import json 

# Opening JSON file
dct_dataframe = {'text': [], 'intent': []}

f = open('./taboo_data/robust_data_collection/intent_classification/taboo_1_config.json')

# returns JSON object as 
# a dictionary
data = json.load(f)

# Closing file
f.close()

dct_phrases = {}
dct_taboo = {}
for key in data:
    dct_phrases[key] = data[key]['phrases']
    dct_taboo[key] = data[key]['avoid_words']
	
default_prompt = """You are an assistant that earns a living through creating paraphrases.
User: Rephrase an original question or statement 5 times. Original phrase: "{}".
Assistant:
"""

dct_final_prompts = {}

for key in dct_phrases:
    dct_final_prompts[key] = []
    for phrase in dct_phrases[key]:
        dct_final_prompts[key].append((default_prompt.format(phrase), phrase))
		
def request_response_from_falcon(prompt):
    device = "cuda:0"
    outputs_sent = []
    encoding = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.inference_mode():
      outputs = model.generate(
          input_ids = encoding.input_ids,
          attention_mask = encoding.attention_mask,
          generation_config = generation_config
      )
      for output in outputs:
        outputs_sent.append(tokenizer.decode(output, skip_special_tokens=True))
    return outputs_sent
	
dct_responses = {}

for key in dct_final_prompts:
    print(key)
    dct_responses[key] = []
    for prompt in dct_final_prompts[key]:
        print(prompt)
        response = request_response_from_falcon(prompt[0])
        dct_responses[key].append((response), prompt[1])
		
with open('diversity_experiments/collected_responses_in_pkl/falcon-40b/responses_0.pkl', 'wb') as file:
    pickle.dump(dct_responses, file)
    
dct_responses = {}

for key in dct_final_prompts:
    print(key)
    dct_responses[key] = []
    for prompt in dct_final_prompts[key]:
        #print(prompt)
        response = request_response_from_falcon(prompt)
        dct_responses[key].append(response)
		
with open('diversity_experiments/collected_responses_in_pkl/falcon-40b/responses_1.pkl', 'wb') as file:
    pickle.dump(dct_responses, file)
    
dct_responses = {}

for key in dct_final_prompts:
    print(key)
    dct_responses[key] = []
    for prompt in dct_final_prompts[key]:
        #print(prompt)
        response = request_response_from_falcon(prompt[0])
        dct_responses[key].append((response), prompt[1])
		
with open('diversity_experiments/collected_responses_in_pkl/falcon-40b/responses_2.pkl', 'wb') as file:
    pickle.dump(dct_responses, file)
    
default_prompt = """You are an assistant that earns a living through creating paraphrases.
User: Rephrase an original question or statement 5 times. Original phrase: "{}".  Don’t use the words “{}”, "{}", or “{}” in your responses.
Assistant:
"""

dct_final_prompts = {}

for key in dct_phrases:
    dct_final_prompts[key] = []
    for phrase in dct_phrases[key]:
        dct_final_prompts[key].append((defaul_taboo_prompt.format(phrase, dct_taboo[key][0], dct_taboo[key][1], dct_taboo[key][2]), phrase))
        
dct_responses = {}

for key in dct_final_prompts:
    print(key)
    dct_responses[key] = []
    for prompt in dct_final_prompts[key]:
        #print(prompt)
        response = request_response_from_falcon(prompt[0])
        dct_responses[key].append((response), prompt[1])
		
with open('diversity_experiments/collected_responses_in_pkl/falcon-40b/responses_taboo_1.pkl', 'wb') as file
    pickle.dump(dct_responses, file)
    
# Opening JSON file
dct_dataframe = {'text': [], 'intent': []}

f = open('./taboo_data/robust_data_collection/intent_classificationw/taboo_2_config.json')

# returns JSON object as 
# a dictionary
data = json.load(f)

# Closing file
f.close()

dct_phrases = {}
dct_taboo = {}
for key in data:
    dct_phrases[key] = data[key]['phrases']
    dct_taboo[key] = data[key]['avoid_words']

default_prompt = """You are an assistant that earns a living through creating paraphrases.
User: Rephrase an original question or statement 5 times. Original phrase: "{}". Don’t use the words “{}”, "{}", "{}", "{}", "{}" or “{}” in your responses.
Assistant:
"""

dct_final_prompts = {}

for key in dct_phrases:
    dct_final_prompts[key] = []
    for phrase in dct_phrases[key]:
        dct_final_prompts[key].append((default_prompt.format(phrase, dct_taboo[key][0], dct_taboo[key][1], dct_taboo[key][2], dct_taboo[key][3], dct_taboo[key][4], dct_taboo[key][5]), phrase))
        
dct_responses = {}

for key in dct_final_prompts:
    print(key)
    dct_responses[key] = []
    for prompt in dct_final_prompts[key]:
        #print(prompt)
        response = request_response_from_falcon(prompt[0])
        dct_responses[key].append((response), prompt[1])
		
with open('diversity_experiments/collected_responses_in_pkl/falcon-40b/responses_taboo_2.pkl', 'wb') as file
    pickle.dump(dct_responses, file)
#!/usr/bin/env python
# coding: utf-8
"""
This project fine-tunes a small LLM to compete with a larger, more complex, model on summarizing customer feedback. 
The first part of the code uses GPT3.5 to simulate customer feedback messages and their summaries for fine-tuning. 
A 🤗 dataset (see [Datasets](https://github.com/huggingface/datasets)) is created from these generated comments and summaries.
The prompt given to GPT to generate `n_samples` for training and evaluation is
See README.md for details.
"""
import os
import shutil
import uuid
import yaml
import json
import jsonlines
from dotenv import load_dotenv
from datasets import DatasetDict, load_dataset
from openai import OpenAI
from utils import deduplication_jsonlines, clean_jsonlines, jsonl_to_json

# Load configs
configs = yaml.safe_load(open('configs.yaml'))
n_samples = configs['n_samples']
n_runs = configs['n_runs']
data_fp = configs['data_fp']
data_dir = configs['data_dir']
tar_fp = configs['tar_fp']
generation_model = configs['generation_model']

print(f"Generating {n_samples} samples for {n_runs} runs using {generation_model} model")

load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=OPENAI_API_KEY)

class CustomerFeedback:
    """
    The `CustomerFeedback` class provides methods to load, generate (simulated with GPT3.5), and save customer feedback samples. It can be used to create a dataset of simulated customer feedback for training or evaluation purposes.
    
    The `load_samples` method loads existing feedback samples from a JSON file. The `generate_samples` method uses the GPT-3.5 language model to generate new feedback samples and summaries. The `save_samples` method saves the generated samples to a JSON file.
    
    The class also provides methods to access the samples, such as `__len__`, `__getitem__`, and `__iter__`.
    """
    def __init__(self, fp):
        self.fp = fp
        self.samples = []
        self.prompt = None

    def load_samples(self):
        samples = []
        if os.path.exists(self.fp):
            with jsonlines.open(self.fp) as reader:
                for obj in reader:
                    samples.append(obj)
        self.samples.extend(samples)
        print(f"Loaded {len(samples)} samples from {self.fp}")

    def _generate_samples(self, n_samples, model, temperature=1.5, max_tokens=50):
        """Generate samples using (default) GPT-3.5
            Parameters
            ----------
            n_samples : int
                Number of samples to generate
            model : str
                Model to use for generation
            temperature : float
                Temperature to use for generation, closer to 2 makes output more unique
            max_tokens : int
                Maximum number of output tokens (not used)

            Returns
            -------
            feedback : list
                List of generated samples
        """
        prompt = "I need your help simulating a customer feedback interaction. "
        prompt += f"Please generate n={n_samples} passages of one to two paragraph from "
        prompt += "a customer to a large insurance company (\"Everywhere Insurance Co.\"). "
        prompt += "Customers can ask questions or lob complaints or give feedback about autos "
        prompt += "(cars, trucks, boats, RVs, motorcycles) or property (houses, apartments, condos). "
        prompt += "Also return your own summary of the text you generated. Structure the output as "
        prompt += "{'summary': <summary of feedback>, 'feedback':<text of feedback>}. "
        prompt += "Make sure it is well-formatted JSON!"
        self.prompt = prompt
        response = client.chat.completions.create(
            model=model,
            response_format={"type":"json_object"},
            temperature=temperature, # default=1, more randomness, must be <= 2
            messages=[{"role": "system", "content": "You are an AI that generates structured output for building training and eval datasets."}, {"role": "user", "content": prompt}],
            #max_tokens=max_tokens,
            )
        try:
            content = eval(response.choices[0].message.content)
            key = list(content.keys())[0]
            feedback = content[key]
        except:
            feedback = {"summary":"", "feedback":""}
        
        return feedback

    def generate_samples(self, n_samples, model="gpt-3.5-turbo-0125", save=True):        
        samples = self._generate_samples(n_samples=n_samples, model=model)
        self.samples = samples
        if save:
            self.save_samples()
        return samples

    def save_samples(self):
        with jsonlines.open(self.fp, 'a') as writer:
            for sample in self.samples:
                writer.write(sample)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def __iter__(self):
        return iter(self.samples)

    def __repr__(self):
        return f"CustomerFeedback({self.fp})"

data_samples = CustomerFeedback(fp=data_fp)
for i in range(n_runs):
    data_samples.generate_samples(n_samples=n_samples, model=generation_model, save=True)
    print(i, len(data_samples))

deduplication_jsonlines(data_fp)
clean_jsonlines(infile=data_fp)
print(data_fp, len(open(data_fp, 'r').readlines()))

# add a unique id to each line in jsonl file
output_fp = 'data/_data.jsonl'
with open(data_fp, 'r') as infile, open(output_fp, 'w') as outfile:
    for line in infile:
        data = json.loads(line)
        data['id'] = str(uuid.uuid4())
        outfile.write(json.dumps(data) + '\n')
os_command = f"mv {output_fp} {data_fp}"
print(os_command, os.system(os_command))

# Convert jsonlines file to json blob
data_fp_json = data_fp.replace(".jsonl", ".json")
jsonl_to_json(data_fp, data_fp_json)

# Load the dataset
data_fp_json = 'data/data.json'
dataset = load_dataset('json', data_files=data_fp_json, split='train')

# Perform an initial split for train and test
train_test_split = dataset.train_test_split(test_size=0.2)  # 80% train, 20% for test
train_dataset = train_test_split['train']
test_dataset = train_test_split['test']

# Split the test dataset further into test and validation
test_val_split = test_dataset.train_test_split(test_size=0.5)  # Split the 20% into two parts of 10% each
test_dataset = test_val_split['test']
validation_dataset = test_val_split['train']

# Combine into a DatasetDict
dataset_dict = DatasetDict({
    'train': train_dataset,
    'validation': validation_dataset,
    'test': test_dataset
})

# save and push to HF hub
print(dataset_dict)
dataset_dict.save_to_disk('data/hf_customer_feedback')
print(dataset_dict)

shutil.make_archive(base_name=tar_fp[:-7], format='gztar', root_dir=data_dir)
print(f"Created {tar_fp} from {data_dir}")
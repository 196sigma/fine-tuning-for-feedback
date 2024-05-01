# Fine-tuning a Language Model to Summarize Customer Feedback

[Reginald Edwards](https://www.reginaldedwards.com) | [LinkedIn](https://www.linkedin.com/in/reginald-edwards) | [Follow me on X](https://www.twitter.com/lereggie)

This project fine-tunes a small LLM to compete with a larger, more complex, model on summarizing customer feedback. 
The first part of the code uses GPT3.5 to simulate customer feedback messages and their summaries for fine-tuning. 
A 🤗 dataset (see [Datasets](https://github.com/huggingface/datasets)) is created from these generated comments and summaries.
The prompt given to GPT to generate `n_samples` for training and evaluation is

```
I need your help simulating a customer feedback interaction. Please generate 10 one to two paragraph passages from a customer to a large insurance company ("Everywhere Insurance Co."). Customers can ask questions or lob complaints or give feedback about autos (cars, trucks, boats, RVs, motorcycles) or property (houses, apartments, condos). Also return your own summary of the text you generated. Structure the output as {{'summary': <summary of feedback>, 'feedback':<text of feedback>}}. Make sure it is well-formatted JSON!
```

Fed into the OpenAI API as

```Python
messages=[
    {
        "role": "system", 
        "content": "You are an AI that generates structured output for building training and eval datasets."
    },
    {
        "role": "user", 
        "content": f"""I need your help simulating a customer feedback interaction. Please generate {n_samples} one to two paragraph passages from a customer to a large insurance company ("Everywhere Insurance Co."). Customers can ask questions or lob complaints or give feedback about autos (cars, trucks, boats, RVs, motorcycles) or property (houses, apartments, condos). Also return your own summary of the text you generated. Structure the output as {{'summary': <summary of feedback>, 'feedback':<text of feedback>}}. Make sure it is well-formatted JSON!"""
    },
]
```

An example training sample is then:
```JSON
{
    "summary": "Customer with a complaint about a denied auto insurance claim", 
    "feedback": "Hello Everywhere Insurance Co., I am writing to express my disappointment regarding my recent auto insurance claim. My car was involved in a minor fender bender, and when I reached out to file the claim, I was shocked to learn that it was denied due to a technicality in the policy. I have been a loyal customer for many years, and I expect better support and service in times of need. I would appreciate a thorough review of my case and a reconsideration of the decision.", 
    "id": "b1d2efe4-d4c1-42ed-9e8b-d40f7ff389d2"
},
{
    "summary": "Geneal praise for customer service",
    "feedback": "I want to take a moment to express my appreciation for the exceptional customer service provided by Everywhere Insurance Co. The representatives I have interacted with have been knowledgeable, responsive, and empathetic in addressing my inquiries and concerns. Kudos to Everywhere Insurance Co. for prioritizing customer satisfaction.",
    "id": "6f0b57d3-1ad4-4078-a9d4-4e4f3fc97735"
},
...
```

The model selected to fine-tune is the [`t5-small`](https://huggingface.co/google-t5/t5-small) variant of T5 from Google.
[T5](https://github.com/google-research/text-to-text-transfer-transformer) is a sequence-to-sequence Transformer model that was trained on a variety of tasks including summarization, question answering, and translation.
After fine-tuning on the customer feedback data `t5-small` is compared to the base model without fine-tuning and [`t5-base`](https://huggingface.co/google-t5/t5-base) a larger version of the same model.
The three models--`t5-small-finetuned`, `t5-small`, and `t5-base` are then evaluated on a held out set of feedback samples and compared on [Rouge](https://huggingface.co/spaces/evaluate-metric/rouge) and [BLEU](https://huggingface.co/spaces/evaluate-metric/bleu) metrics.

This project requires `Python 3.11` and the fine-tuning is best done in [Colab](colab.research.google.com) with a GPU/TPU attached.

## Setup
Create a new environment via conda or similar:

`conda create -name feedback_env python=3.11`

Install the project and requirements

`pip install -r requirements.txt`

`pip install -e .`


## Data generation
Specify important parameters in `notebooks/configs.yaml` file. 
Samples can be generated in `notebooks/generate_feedback.ipynb` 
or via script `notebooks/generate_feedback.py`.

## Fine-tuning
Run the fine-tuning notebook `notebooks/finetune_feedback.ipynb` with a GPU instance attached. 
On a A100 training takes approximately 1 minute per 10 epochs. 
Inference time on a new sample with a GPU is approximately 2 seconds per sample.

## Evaluation

The evaluation metrics comparing the three models are below. 
The improvement from fine-tuning `t5-small` can be seen in improved Rouge and BLEU scores.

### Rouge

| Model              |   rouge |
|:-------------------|--------:|
| t5-base            |   0.177 |
| t5-small           |   0.174 |
| t5-small-finetuned |   0.149 |

### BLEU

| Model              |   bleu |
|:-------------------|-------:|
| t5-base            |  0.02  |
| t5-small           |  0.024 |
| t5-small-finetuned |  0.022 |

### Full results

| Model                                        | bleu   | rouge   |
|:---------------------------------------------|:-------|:--------|
| ('t5-base', 'bleu')                          | 0.02   | -       |
| ('t5-base', 'brevity_penalty')               | 1.0    | -       |
| ('t5-base', 'length_ratio')                  | 3.721  | -       |
| ('t5-base', 'reference_length')              | 985.0  | -       |
| ('t5-base', 'rouge1')                        | -      | 0.177   |
| ('t5-base', 'rouge2')                        | -      | 0.058   |
| ('t5-base', 'rougeL')                        | -      | 0.144   |
| ('t5-base', 'rougeLsum')                     | -      | 0.144   |
| ('t5-base', 'translation_length')            | 3665.0 | -       |
| ('t5-small', 'bleu')                         | 0.024  | -       |
| ('t5-small', 'brevity_penalty')              | 1.0    | -       |
| ('t5-small', 'length_ratio')                 | 2.97   | -       |
| ('t5-small', 'reference_length')             | 985.0  | -       |
| ('t5-small', 'rouge1')                       | -      | 0.174   |
| ('t5-small', 'rouge2')                       | -      | 0.061   |
| ('t5-small', 'rougeL')                       | -      | 0.146   |
| ('t5-small', 'rougeLsum')                    | -      | 0.146   |
| ('t5-small', 'translation_length')           | 2925.0 | -       |
| ('t5-small-finetuned', 'bleu')               | 0.022  | -       |
| ('t5-small-finetuned', 'brevity_penalty')    | 1.0    | -       |
| ('t5-small-finetuned', 'length_ratio')       | 4.059  | -       |
| ('t5-small-finetuned', 'reference_length')   | 985.0  | -       |
| ('t5-small-finetuned', 'rouge1')             | -      | 0.149   |
| ('t5-small-finetuned', 'rouge2')             | -      | 0.049   |
| ('t5-small-finetuned', 'rougeL')             | -      | 0.128   |
| ('t5-small-finetuned', 'rougeLsum')          | -      | 0.127   |
| ('t5-small-finetuned', 'translation_length') | 3998.0 | -       

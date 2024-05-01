from setuptools import setup, find_packages

setup(
    name='fine_tuning_for_feedback',  
    version='0.1.0', 
    packages=find_packages(), 
    description='Fine-tuning LLMs to summarize customer feedback.',
    author='Reginald Edwards',  
    author_email='reginald.edwards@hey.com', 
    install_requires=[
        'nltk',
        'tabulate',
        'numpy',
        'pandas',
        'notebook',
        'torch',
        'openai',
        'scipy',
        'jsonlines',
        'python-dotenv',
        'datasets',
        'evaluate',
        'accelerate==0.21.0',
        'peft==0.4.0',
        'bitsandbytes==0.40.2',
        'transformers==4.31.0',
        'trl==0.4.7',
    ],
    python_requires='>=3.10, <3.12'
)

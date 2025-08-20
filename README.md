# Masters Thesis Baris Coslu: Gradientless Optimization for Language Model Finetuning

## Abstract

Transformer models pre-trained on large amounts of data have become the standard method of building powerful language models in Natural Language Processing (NLP) research. These models incorporate comprehensive information about language, allowing them to be fine-tuned for a wide variety of tasks using a smaller, task-specific dataset without substantial modifications on the architecture. This work aims to explore gradientless optimization methods that can be utilized in the context of language model fine-tuning. In particular, it investigates the feasibility of using direct search methods based on random perturbations of parameter tensors as an alternative to state-of-the-art first-order optimizers for the fine-tuning of pre-trained language models. We introduce a direct search method based on an adaptation of the Gradientless Descent (GLD) algorithm. Our method can fine-tune a DistilBERT model on the SST-2 dataset using less memory than an Adam optimizer in exchange for a small reduction in validation accuracy.

## Installation
1. Make sure to have [Python](https://www.python.org/) and [pip](https://pypi.org/project/pip/) installed.
2. Create and activate a new virtual environment by changing to the `code` directory and running:  
```
python -m venv venv
```
```
venv/Scripts/activate
```
3. Install the required packages by running: 
```
pip install -r requirements.txt
```
4. Install [PyTorch](https://pytorch.org/) by running the OS and CUDA version-specific installation command.

## Running Experiments
`GldSearchParameter` and its subclasses describe a trainable tensor. A documentation of its attributes can be found in `gld_search_parameter.py`. Most training arguments are set in `training_arguments.py`. This file also sets the default values for the attributes of 
`GldSearchParameter`. To reproduce experiments: 

1. Set the training arguments in `training_arguments.py`.
2. In `main.py`, set the variable `trainable_params` to 
a list of trainable tensors of type `GldSearchParameter`.
3. In `main.py`, adapt the variable `train_args` to set the batch size, number of steps, etc.
4. Run `main.py`.

Experiment results are logged to TensorBoard and can be viewed by running:

```
tensorboard --logdir output/runs
```

## System Setup
All of our experiments have been run on a machine with the following specifications:  

* AMD Ryzen 7 5800H CPU, 3201 Mhz, 8 Cores, 16 Threads
* NVIDIA GeForce RTX 3050Ti Laptop GPU, 4 GB VRAM
* 16 GB RAM 
* Windows 10 Home, Version 22H2
* CUDA Version 12.3

 We used Python version 3.11.3 and PyTorch version 2.2.0+cu121. For other required packages, the respective version can be found in `code/requirements.txt`.

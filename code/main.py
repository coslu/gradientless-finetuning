from evaluate import load
from transformers import AutoModelForSequenceClassification, set_seed, DistilBertForSequenceClassification, \
    RobertaForSequenceClassification
from transformers import TrainingArguments, AutoTokenizer, DataCollatorWithPadding
from transformers.integrations import TensorBoardCallback
from datasets import load_dataset
from numpy import argmax
from gld_search_trainer import GldSearchTrainer
from gld_search_parameter import AllParameter, RandomParameter, AbsParameter, FisherParameter, GldSearchParameter
from models import distilbert_for_sequence_classification, roberta_for_sequence_classification
from functools import partial
from torch.cuda import memory_stats


def tokenize_function(examples):
    return tokenizer(examples['sentence'], truncation=True)


def compute_metrics(eval_prediction):
    metric = load('accuracy')
    logits, labels = eval_prediction
    accuracy = metric.compute(predictions=argmax(logits, axis=-1), references=labels)
    return accuracy


class MemoryCallBack(TensorBoardCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        logs['memory_active_peak'] = memory_stats()['active_bytes.all.peak']
        super().on_log(args, state, control, logs, **kwargs)


if __name__ == '__main__':
    set_seed(42)

    # Select the base model
    model_name = 'distilbert-base-cased'

    # Override the forward function for a faster forward run that can skip the base model
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    if isinstance(model, DistilBertForSequenceClassification):
        model.forward = partial(distilbert_for_sequence_classification.forward, model)
    elif isinstance(model, RobertaForSequenceClassification):
        model.forward = partial(roberta_for_sequence_classification.forward, model)

    # Set training arguments for the base Trainer class
    train_args = TrainingArguments('output', save_strategy='steps', save_steps=5000, save_total_limit=1,
                                   logging_steps=1000, evaluation_strategy='steps', eval_steps=5000,
                                   per_device_train_batch_size=64, per_device_eval_batch_size=64, max_steps=80000)

    # Load dataset
    dataset = load_dataset('sst2')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer)

    # Give the tensors to train as a list of GldSearchParameter
    trainable_params = [
        RandomParameter(model.pre_classifier.weight, sampling_size=768 * 2),
        AllParameter(model.pre_classifier.bias),
        AllParameter(model.classifier.weight),
        AllParameter(model.classifier.bias)
    ]

    # Begin training
    trainer = GldSearchTrainer(model=model, args=train_args, train_dataset=tokenized_datasets['train'],
                               eval_dataset=tokenized_datasets['validation'], data_collator=data_collator,
                               compute_metrics=compute_metrics, trainable_params=trainable_params,
                               callbacks=[MemoryCallBack()])
    trainer.train()

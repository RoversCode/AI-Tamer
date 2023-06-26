#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   augment.py
@Time    :   2023/05/31 17:28:22
@Author  :   ChengHee
@Version :   1.0
@Contact :   1059885524@qq.com
@Desc    :   None
"""

# here put the import lib
from transformers import HfArgumentParser, TrainingArguments
from dataclasses import dataclass, field


@dataclass
class DataTrainingArguments:
    train_path: str = field(
        default="nlp/text2text_generation/data/AdvertiseGen/train.json",
        metadata={"help": "Path to training data."},
    )
    eval_path: str = field(
        default="nlp/text2text_generation/data/AdvertiseGen//dev.json",
        metadata={"help": "Path to test data."},
    )
    max_seq_length: int = field(
        default=128,
        metadata={"help": "The maximum total input sequence length after tokenization."},
    )


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default=r"pretrained_models/nlp/mt5-base",  
        metadata={
            "help": "来自huggingface.co/models的预训练模型名或模型存储路径"
        },
    )
    model_max_length: str = field(
        default=512,
        metadata={
            "help": "模型最大输入长度"
        },
    )
    


def get_args():
    """Parse all the args."""
    # parser = HfArgumentParser(
    #     (ModelArguments, DataTrainingArguments, TrainingArguments)
    # )
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments)
    )
    training_args = TrainingArguments(
        output_dir='nlp/text2text_generation/output/',
        num_train_epochs=10,
        seed=42,
        per_device_train_batch_size=8,
        logging_steps=10,
        do_eval=True,
        evaluation_strategy="steps",
        eval_steps=2000,
    )
    args = parser.parse_args_into_dataclasses()
    args = list(args)
    args.append(training_args)
    return tuple(args)


if __name__ == "__main__":
    args = get_args()
    _, data_args, training_args = args
    print(training_args)

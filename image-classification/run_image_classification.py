from dataclasses import dataclass, field
from typing import Optional

import datasets
import numpy as np
from datasets import load_dataset
from transformers import (
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    AutoFeatureExtractor,
    AutoModelForImageClassification,
    MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING
)
from utils import ImageClassificationCollator, image_loader


MODEL_CONFIG_CLASSES = list(MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    dataset_name: Optional[str] = field(
        default="nateraw/image-folder", metadata={"help": "Name of a dataset from the datasets package"}
    )
    train_folder: Optional[str] = field(
        default=None, metadata={"help": "A folder containing the training data."}
    )
    validation_folder: Optional[str] = field(
        default=None, metadata={"help": "A folder containing the validation data."}
    )
    test_folder: Optional[str] = field(
        default=None, metadata={"help": "A folder containing the test data."}
    )
    train_val_split: Optional[float] = field(
        default=0.15, metadata={"help": "Percent to split off of train for validation."}
    )

    @property
    def data_files(self):
        data_files = None
        if self.train_folder is not None:
            data_files = {"train": self.train_folder}
        if self.validation_folder is not None:
            data_files["val"] = self.validation_folder
        if self.test_folder is not None:
            data_files["test"] = self.test_folder
        return data_files


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    ds = load_dataset(
        data_args.dataset_name,
        data_files=data_args.data_files,
        cache_dir=model_args.cache_dir,
        task='image-classification'
    ).with_transform(image_loader)

    if isinstance(data_args.train_val_split, float) and data_args.train_val_split > 0.0:
        split = ds["train"].train_test_split(0.15)
        ds['train'] = split['train']
        ds['validation'] = split['test']


    labels = ds["train"].features["labels"].names
    label2id, id2label = dict(), dict()
    for i, label in enumerate(labels):
        label2id[label] = str(i)
        id2label[str(i)] = label

    def model_init():
        return AutoModelForImageClassification.from_pretrained(
            model_args.model_name_or_path,
            num_labels=len(labels),
            label2id=label2id,
            id2label=id2label,
        )

    feature_extractor = AutoFeatureExtractor.from_pretrained(model_args.model_name_or_path)
    collator = ImageClassificationCollator(feature_extractor)

    metric = datasets.load_metric("accuracy")
    def compute_metrics(p):
        return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)

    trainer = Trainer(
        model_init=model_init,
        args=training_args,
        train_dataset=ds["train"] if training_args.do_train else None,
        eval_dataset=ds["validation"] if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=feature_extractor,
        data_collator=collator,
    )

    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=None)
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()

    if training_args.do_eval:
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


    if training_args.push_to_hub:
        kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "image-classification"}
        trainer.push_to_hub(**kwargs)


if __name__ == "__main__":
    main()

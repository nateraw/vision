# coding=utf-8
# Copyright 2018 HuggingFace Inc..
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import argparse
import json
import logging
import os
import sys
from unittest.mock import patch

import torch

from transformers.file_utils import is_apex_available
from transformers.testing_utils import TestCasePlus, get_gpu_count, slow, torch_device


SRC_DIRS = [
    os.path.join(os.path.dirname(__file__), dirname)
    for dirname in [
        "image-classification",
    ]
]
sys.path.extend(SRC_DIRS)


if SRC_DIRS is not None:
    import run_image_classification
    # import run_clm
    # import run_generation
    # import run_glue
    # import run_mlm
    # import run_ner
    # import run_qa as run_squad
    # import run_summarization
    # import run_swag
    # import run_translation


logging.basicConfig(level=logging.DEBUG)

logger = logging.getLogger()


def get_setup_file():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f")
    args = parser.parse_args()
    return args.f


def get_results(output_dir):
    results = {}
    path = os.path.join(output_dir, "all_results.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            results = json.load(f)
    else:
        raise ValueError(f"can't find {path}")
    return results


def is_cuda_and_apex_available():
    is_using_cuda = torch.cuda.is_available() and torch_device == "cuda"
    return is_using_cuda and is_apex_available()


class ExamplesTests(TestCasePlus):

    def test_run_image_classification(self):
        stream_handler = logging.StreamHandler(sys.stdout)
        logger.addHandler(stream_handler)

        tmp_dir = self.get_auto_remove_tmp_dir()
        testargs = f"""
            run_image_classification.py
            --output_dir {tmp_dir}
            --model_name_or_path google/vit-base-patch16-224-in21k
            --dataset_name nateraw/beans
            --do_train
            --do_eval
            --learning_rate 2e-5
            --per_device_train_batch_size 2
            --per_device_eval_batch_size 1
            --remove_unused_columns False
            --overwrite_output_dir True
            --dataloader_num_workers 16
            --metric_for_best_model accuracy
            --max_steps 30
            --seed 7
        """.split()

        if is_cuda_and_apex_available():
            testargs.append("--fp16")

        with patch.object(sys, "argv", testargs):
            run_image_classification.main()
            result = get_results(tmp_dir)
            self.assertGreaterEqual(result["eval_accuracy"], 0.7)

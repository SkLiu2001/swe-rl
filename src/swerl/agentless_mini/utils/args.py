# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.

from dataclasses import dataclass, field
from typing import Literal

from datasets import Dataset, load_dataset,load_from_disk
from transformers.hf_argparser import DataClassType, HfArgumentParser


@dataclass(frozen=True)
class BenchArgs:
    shard: int = field(metadata={"help": "The shard to process"})
    num_shards: int = field(metadata={"help": "The total number of shards"})
    # We've only tested on the Verified subset currently,
    # but you can use shard/num_shards to split the dataset
    dataset: Literal[
        "princeton-nlp/SWE-bench_Lite", "princeton-nlp/SWE-bench_Verified"
    ] = field(default="princeton-nlp/SWE-bench_Verified")
    local_dataset_path: str = field(
        default="",
        metadata={"help": "Local path to the dataset if not loading from Hugging Face Hub"}
    )

    def load(self) -> Dataset:
        if self.local_dataset_path:
            dataset = load_from_disk(self.local_dataset_path)
        else:
            dataset = load_dataset(self.dataset, split="test")
        return dataset.shard(
            num_shards=self.num_shards,
            index=self.shard,
            contiguous=False,
        )


@dataclass(frozen=True)
class InferenceArgs:
    model: str = field(
        metadata={"help": "The model id sent to the openai inference backend"}
    )
    temperature: float
    num_samples: int
    max_tokens: int = field(default=4096)
    max_concurrent_requests: int = field(
        default=64,
        metadata={"help": "Maximum number of concurrent requests sent to the backend"},
    )
    retries: int = field(default=3, metadata={"help": "Number of retries for failed requests"})
    delay: int = field(default=30, metadata={"help": "Delay between retries"})

def parse_args_into_dataclasses(*classes: DataClassType):
    return HfArgumentParser(classes).parse_args_into_dataclasses()

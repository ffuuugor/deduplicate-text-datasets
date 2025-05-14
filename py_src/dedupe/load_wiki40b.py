# Copyright 2021 Google LLC
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# import tensorflow_datasets as tfds
# import tensorflow as tf
import os
import struct
import numpy as np
from transformers import GPT2Tokenizer, T5Tokenizer
import multiprocessing as mp
from datasets import load_dataset, load_from_disk
import datasets
from tqdm import tqdm

import argparse

parser = argparse.ArgumentParser(description="Load a dataset.")
parser.add_argument("--save-dir", type=str)
parser.add_argument("--parts", type=int)
parser.add_argument("--name", type=str)
parser.add_argument("--tokenize", action="store_true")
parser.add_argument("--tokenizer", type=str, default="gpt2")
parser.add_argument("--pre-sep", type=bytes, default=b"\xff\xff")
parser.add_argument("--post-sep", type=bytes, default=b"")
parser.add_argument("--num-proc", type=int, default=80)
args = parser.parse_args()

if args.tokenize:
    if args.tokenizer == "gpt2":
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    elif args.tokenizer == "t5":
        tokenizer = T5Tokenizer.from_pretrained("t5-small")
    else:
        raise

# datasets.disable_progress_bars()

# BASE_PATH="train/chunk{chunk}/example_train_{i}.jsonl.zst"
BASE_PATH="/data/igor/wiki40b/en/test-0000{i}-of-00002.parquet"

parts_per_chunk = 1

for part in range(args.parts):
    files = [BASE_PATH.format(i=part)]

    print(files)
    ds = load_dataset("google/wiki40b", data_files=files, num_proc=1)

    split = "train"
    save_dir = args.save_dir
    dataset_name = f"{args.name}_{part}_of_{args.parts}"
    pre_sep = args.pre_sep
    post_sep = args.post_sep

    ds = ds[split]

    UID = 0

    def sep():
        global UID
        UID += 1
        return pre_sep + struct.pack("<I", UID) + post_sep

    def tok(x):
        if args.tokenize:
            out = tokenizer.encode(x["text"])
            out = np.array(out, dtype=np.uint16).view(np.uint8).tobytes()
            return {"bytes": out}
        else:
            out = x
        return out

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    path = os.path.join(save_dir, dataset_name + "." + split)
    print("OLOLO", path)
    fout = open(os.path.join(save_dir, dataset_name + "." + split), "wb")

    slice_size = 1_000_000

    i = 0
    sizes = [0]

    for i in tqdm(range(0, len(ds), slice_size)):
        ds2 = ds.select(range(i, min(i + slice_size, len(ds))))
        ds2 = ds2.map(
            tok,
            num_proc=64,
            remove_columns=ds.column_names,
        )

        for text in ds2["bytes"]:
            next_line = sep() + text
            fout.write(next_line)
            sizes.append(sizes[-1] + len(next_line))
            i += 1

    open(os.path.join(save_dir, dataset_name + "." + split + ".size"), "wb").write(
        np.array(sizes, dtype=np.uint64).tobytes()
    )

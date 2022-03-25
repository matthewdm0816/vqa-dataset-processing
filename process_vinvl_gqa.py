import ast
import base64
import csv
import enum
import glob
import json
import logging
import os
import os.path as osp
import pickle
from re import L
import sys
from collections import defaultdict
from threading import Thread

import colorama
import datasets
import numpy as np
import pandas as pd
import toml
import torch
import transformers
from datasets import Dataset, load_dataset
from icecream import ic
from tqdm import *
from tqdm.auto import tqdm
import argparse
from more_itertools import chunked


def mapdict(f, d):
    for k, v in d.items():
        print(k, v)
    return {k: f(v) if isinstance(v, str) else list(map(f, v)) for k, v in d.items()}


def _getid(image_path):
    r"""
    "VizWiz-werwerwer-0001123.jpg" => 1123
    "COCO_train2014_00000000123123.jpg" => 123123
    """
    filename = os.path.basename(image_path)
    image_id = filename.split("_")[-1]
    image_id = image_id.split(".")[0]
    return int(image_id)


def getiid(image_id_str: str):
    try:
        # NOTE: for gqa, use string image_id
        #   since there are image_ids like 'n1234'
        image_id = image_id_str
    except ValueError:
        ic(image_id_str)
        image_id = _getid(image_id_str)
    return image_id


def readvinvl_line(line: str):
    image_id, num_boxes, feature_enc = line[0], int(line[1]), line[-1]
    # try:
    #     image_id = int(image_id)
    # except ValueError:
    #     # not int but image name
    #     image_id = _getid(image_id)
    image_id = getiid(image_id)
    decoded = np.frombuffer(base64.b64decode(feature_enc), np.float32).reshape(
        num_boxes, -1
    )
    return image_id, num_boxes, decoded


def readvinvl_lines(lines: list[str]):
    res = []
    for line in lines:
        image_id, num_boxes, feature_enc = readvinvl_line(line)
        res.append(
            {
                "image_id": image_id,
                "num_boxes": num_boxes,
                "feature_enc": feature_enc,
            }
        )
    return res


parser = argparse.ArgumentParser()
parser.add_argument(
    "--feature_path", type=str, default="~/data/vinvl_gqa/model_0060000"
)
parser.add_argument(
    "--target_path",
    type=str,
    default="~/data/vinvl_gqa/standalone_feature_vinvl_gqa.pkl",
)
parser.add_argument(
    "--dataset_save_path",
    type=str,
    default="~/data/vinvl_gqa/processed_without_features_vinvl_gqa",
)
parser.add_argument(
    "--balanced",
    action="store_true",
    help="If set, only use balanced dataset, please change output path accordingly",
)

args = parser.parse_args()


logger = logging.getLogger()
logger.setLevel(logging.INFO)
log_format = (
    colorama.Fore.MAGENTA
    + "[%(asctime)s %(name)s %(levelname)s] "
    + colorama.Fore.WHITE
    + "%(message)s"
)
logging.basicConfig(format=log_format, level=logging.INFO, datefmt="%I:%M:%S")

logging.info(f"Loading {'balanced' if args.balanced else 'full'} dataset...")
if args.balanced:
    # Load balanced GQA dataset (1.4M)
    dataset = load_dataset(
        "json",
        data_files=mapdict(
            lambda x: osp.expanduser(x),
            {
                "train": "~/data/vinvl_gqa/questions/train_balanced_questions.json.mod",
                "val": "~/data/vinvl_gqa/questions/val_balanced_questions.json.mod",
                # "test": "~/data/vinvl_gqa/questions/test_balanced_questions.json.mod",
                # "test_dev": "~/data/vinvl_gqa/questions/testdev_balanced_questions.json.mod",
                # "challenge": "~/data/vinvl_gqa/questions/challenge_balanced_questions.json.mod",
            },
        ),
        field='data',
    )
    dataset_noanno = load_dataset(
        "json",
        data_files=mapdict(
            lambda x: osp.expanduser(x),
            {
                # "train": "~/data/vinvl_gqa/questions/train_balanced_questions.json.mod",
                # "val": "~/data/vinvl_gqa/questions/val_balanced_questions.json.mod",
                "test": "~/data/vinvl_gqa/questions/test_balanced_questions.json.mod",
                "test_dev": "~/data/vinvl_gqa/questions/testdev_balanced_questions.json.mod",
                "challenge": "~/data/vinvl_gqa/questions/challenge_balanced_questions.json.mod",
            },
        ),
        field='data',
    )

else:
    # Load full GQA dataset (22M)
    dataset = load_dataset(
        "json",
        data_files=mapdict(
            lambda x: osp.expanduser(x),
            {
                "train": glob.glob(
                    osp.expanduser(
                        "~/data/vinvl_gqa/questions/train_all_questions/train_all_questions_*.json.mod"
                    )
                ),
                "val": "~/data/vinvl_gqa/questions/val_balanced_questions.json.mod",
                # "test": "~/data/vinvl_gqa/questions/test_balanced_questions.json.mod",
                # "test_dev": "~/data/vinvl_gqa/questions/testdev_balanced_questions.json.mod",
                # "challenge": "~/data/vinvl_gqa/questions/challenge_balanced_questions.json.mod",
            },
        ),
        field='data',
    )
    dataset_noanno = load_dataset(
        "json",
        data_files=mapdict(
            lambda x: osp.expanduser(x),
            {
                # "train": "~/data/vinvl_gqa/questions/train_balanced_questions.json.mod",
                # "val": "~/data/vinvl_gqa/questions/val_balanced_questions.json.mod",
                "test": "~/data/vinvl_gqa/questions/test_balanced_questions.json.mod",
                "test_dev": "~/data/vinvl_gqa/questions/testdev_balanced_questions.json.mod",
                "challenge": "~/data/vinvl_gqa/questions/challenge_balanced_questions.json.mod",
            },
        ),
        field='data',
    )

for split in ['test', 'test_dev', 'challenge']:
    dataset[split] = dataset_noanno[split]

dataset = dataset.rename_column("imageId", "image_id")

ic(dataset["train"].features)

sorted_dataset = dataset.sort("question_id")
ic(sorted_dataset["train"][:3])

# ic.configureOutput(outputFunction=print)
csv.field_size_limit(sys.maxsize)

feature_paths = list(map(os.path.expanduser, args.feature_path.split(",")))
for feature_path in feature_paths:
    assert os.path.exists(feature_path), f"{feature_path} does not exist"
logging.info("Loading Features from {}".format(feature_paths))
record_features = dict()
record_tags = dict()

for feature_path in feature_paths:
    logging.info(f"Loading features from {feature_path}")
    tsv_path = osp.expanduser(f"{feature_path}/features.tsv")
    pred_path = osp.expanduser(f"{feature_path}/predictions.tsv")
    lidx_path = osp.expanduser(f"{feature_path}/features.lineidx")
    imgid2idx_path = osp.expanduser(f"{feature_path}/imageid2idx.json")

    def transform_dict(d: dict) -> dict:
        keys = d.items()[0].keys()
        ic(keys)
        d_new = defaultdict(list)
        for k, vdict in d.items():
            for k_, v in vdict:
                d_new[k_].append(v)
        return d_new

    logging.info("Loading Features...")
    with open(imgid2idx_path, "rt") as f:
        imgid2idx = json.load(f)
    total_lines = max([idx for imgid, idx in imgid2idx.items()]) + 1
    ic(total_lines)
    # Load all images feature from VinVL TSV | includes test/train/val split
    with open(tsv_path) as tsvfile:
        tsvreader = csv.reader(tsvfile, delimiter="\t")

        # for line in tqdm(tsvreader, total=total_lines):
        #     image_id, num_boxes, feature_enc = line[0], int(line[1]), line[-1]
        #     try:
        #         image_id = int(image_id)
        #     except ValueError:
        #         # not int but image name
        #         image_id = _getid(image_id)
        #     decoded = np.frombuffer(base64.b64decode(feature_enc), np.float32).reshape(
        #         num_boxes, -1
        #     )
        #     # ic(image_id, num_boxes, decoded.shape)
        #     record_features[image_id] = decoded

        # Concurrent version of TSV Reading
        from concurrent.futures import ThreadPoolExecutor, wait

        executor = ThreadPoolExecutor(max_workers=16)
        futures = []

        for line in tqdm(tsvreader, total=total_lines):
            futures.append(executor.submit(readvinvl_line, line))
        wait(futures)
        for future in tqdm(futures):
            image_id, num_boxes, decoded = future.result()
            record_features[image_id] = decoded

        # Multiprocessing version of TSV Reading
        # from multiprocessing import Pool, Queue
        # from concurrent.futures import ProcessPoolExecutor, Future
        # with ProcessPoolExecutor(max_workers=16) as executor:
        #     futures: list[Future] = []
        #     for line in tqdm(tsvreader, total=total_lines):
        #         futures.append(executor.submit(readvinvl_line, line))
        #     for future in tqdm(futures):
        #         image_id, num_boxes, decoded = future.result()
        #         record_features[image_id] = decoded

    logging.info("Loading Tags...")
    with open(pred_path) as tsvfile:
        tsvreader = csv.reader(tsvfile, delimiter="\t")
        for line in tqdm(tsvreader, total=total_lines):
            # print(line)
            object_info = json.loads(line[-1])
            object_tags = list(
                map(
                    lambda object_info_it: object_info_it["class"],
                    object_info["objects"],
                )
            )
            # ic(object_tags)
            # image_id = int(line[0])
            image_id = getiid(line[0])
            record_tags[image_id] = object_tags


logging.info("Parsing GQA Annotation and Questions")
# dataset = load_dataset(
#     "json",
#     field="annotations",
#     data_files={
#         "train": "annotation/v2_mscoco_train2014_annotations.json",
#         "val": "annotation/v2_mscoco_val2014_annotations.json",
#     },
#     keep_in_memory=False,
# )
# ic(dataset)
# dataset_ques = load_dataset(
#     "json",
#     field="questions",
#     data_files={
#         "train": "question/v2_OpenEnded_mscoco_train2014_questions.json",
#         "val": "question/v2_OpenEnded_mscoco_val2014_questions.json",
#         "test": "question/v2_OpenEnded_mscoco_test2015_questions.json",
#         "test_dev": "question/v2_OpenEnded_mscoco_test-dev2015_questions.json",
#     },
#     keep_in_memory=False,
# )
# ic(dataset_ques)


logging.info("Adding Test-only I-Q pairs...")

# Collect all iid-qid pairs
# trainval_qids = set()


# def gather_qids(samples):
#     trainval_qids.update(samples["question_id"])

# sorted_dataset.map(gather_qids, batched=True)

# test_dataset = sorted_dataset_ques.filter(
#     lambda sample: sample["question_id"] not in trainval_qids
# )
# ic(test_dataset, test_dataset["test"].features)

# sorted_dataset["test"] = test_dataset["test"]
# sorted_dataset["test_dev"] = test_dataset["test_dev"]

# Load tags
def batched_load_tags(samples):
    fids = samples["image_id"]
    return {
        "tags": [record_tags[fid] for fid in fids],
    }


# Instead of saving feature in every I-Q pairs, we save a standalone dataset
logging.info("Merging Features...")
sorted_dataset = sorted_dataset.map(batched_load_tags, batched=True)
# feature_dataset = Dataset.from_dict(feature_loader.features_cache)

ic(sorted_dataset["train"]["tags"][0])
logging.info("Saving to Disk...")
sorted_dataset.save_to_disk(osp.expanduser(args.dataset_save_path))
with open(osp.expanduser(args.target_path), "wb") as f:
    pickle.dump(record_features, f)

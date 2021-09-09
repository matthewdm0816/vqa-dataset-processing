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
import sys
from collections import defaultdict

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

logger = logging.getLogger()
logger.setLevel(logging.INFO)
log_format = (
    colorama.Fore.MAGENTA
    + "[%(asctime)s %(name)s %(levelname)s] "
    + colorama.Fore.WHITE
    + "%(message)s"
)
logging.basicConfig(format=log_format, level=logging.INFO, datefmt="%I:%M:%S")

# ic.configureOutput(outputFunction=print)
csv.field_size_limit(sys.maxsize)


tsv_path = osp.expanduser("~/data/vinvl/model_0060000/features.tsv")
pred_path = osp.expanduser("~/data/vinvl/model_0060000/predictions.tsv")
lidx_path = osp.expanduser("~/data/vinvl/model_0060000/features.lineidx")
imgid2idx_path = osp.expanduser("~/data/vinvl/model_0060000/imageid2idx.json")

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
record_features = dict()
with open(tsv_path) as tsvfile:
    tsvreader = csv.reader(tsvfile, delimiter="\t")
    for line in tqdm(tsvreader, total=total_lines):
        image_id, num_boxes, feature_enc = int(line[0]), int(line[1]), line[-1]
        decoded = np.frombuffer(base64.b64decode(feature_enc), np.float32).reshape(
            num_boxes, -1
        )
        # ic(image_id, num_boxes, decoded.shape)
        record_features[image_id] = decoded

logging.info("Loading Tags...")
record_tags = dict()
with open(pred_path) as tsvfile:
    tsvreader = csv.reader(tsvfile, delimiter="\t")
    for line in tqdm(tsvreader, total=total_lines):
        # print(line)
        object_info = json.loads(line[-1])
        object_tags = list(map(lambda object_info_it: object_info_it['class'],object_info['objects']))
        # ic(object_tags)
        image_id = int(line[0])
        record_tags[image_id] = object_tags


logging.info("Parsing Annotation and Questions")
dataset = load_dataset(
    "json",
    field="annotations",
    data_files={
        "train": "annotation/v2_mscoco_train2014_annotations.json",
        "val": "annotation/v2_mscoco_val2014_annotations.json",
    },
    keep_in_memory=False,
)
ic(dataset)
dataset_ques = load_dataset(
    "json",
    field="questions",
    data_files={
        "train": "question/v2_OpenEnded_mscoco_train2014_questions.json",
        "val": "question/v2_OpenEnded_mscoco_val2014_questions.json",
    },
    keep_in_memory=False,
)
ic(dataset_ques)
ic(dataset_ques["train"][0]["question"].split(" "))
items = dataset_ques["train"].filter(
    lambda item: item["question"] == "Is the mountain rocky?"
)
ic(items)
sorted_dataset_ques = dataset_ques.sort("question_id")
ic(sorted_dataset_ques["train"][:3])
sorted_dataset = dataset.sort("question_id")
ic(sorted_dataset["train"][:3])

logging.info("Merging into Q-A Pairs")


def getqid2ques(dataset):
    res = dict()

    def write_qid(samples):
        # for idx, _ in enumerate(samples['question_id']):
        #     res[samples['question_id'][idx]] = samples['question'][idx]
        for qid, q in zip(samples["question_id"], samples["question"]):
            res[qid] = q

    dataset.map(write_qid, batched=True)
    return res


qid2ques = getqid2ques(sorted_dataset_ques)

ic(sorted_dataset_ques.filter(lambda item: item["question_id"] == 25000))
ic(qid2ques[25000])


def add_ques(samples, qid2q):
    return {"question": [qid2q[qid] for qid in samples["question_id"]]}


sorted_dataset = sorted_dataset.map(
    lambda samples: add_ques(samples, qid2ques), batched=True
)
ic(sorted_dataset["train"][3])

sorted_dataset.save_to_disk("processed")

# TODO: Load Features
def batched_load_tags(samples):
    fids = samples["image_id"]
    return {
        "tags": [record_tags[fid] for fid in fids],
    }

# Instead of saving feature in every I-Q pairs, we save a standalone dataset
logging.info("Merging Features...")
sorted_dataset = sorted_dataset.map(batched_load_tags, batched=True)
# feature_dataset = Dataset.from_dict(feature_loader.features_cache)

ic(sorted_dataset['train']['tags'][0])
logging.info("Saving to Disk...")
sorted_dataset.save_to_disk("processed_without_features_vinvl")
with open('standalone_feature_vinvl.pkl', 'wb') as f:
    pickle.dump(record_features, f)

import enum
import os, sys
import datasets
from datasets import load_dataset, Dataset
import json, toml
from icecream import ic
from tqdm import *
import logging
import colorama
import numpy as np
import ast
import glob
import torch
import pickle
logger = logging.getLogger()
logger.setLevel(logging.INFO)
log_format = (
    colorama.Fore.MAGENTA
    + "[%(asctime)s %(name)s %(levelname)s] "
    + colorama.Fore.WHITE
    + "%(message)s"
)
logging.basicConfig(format=log_format, level=logging.INFO, datefmt="%I:%M:%S")

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

region_feature_path = "features/"


class FeatureLoader:
    def __init__(self, path):
        self.path = path
        self.features_cache = dict()

    @classmethod
    def _getid(cls, image_path):
        r"""
        "VizWiz-werwerwer-0001123.jpg" => 1123
        "COCO_train2014_00000000123123.jpg" => 123123
        """
        filename = os.path.basename(image_path)
        image_id = filename.split("_")[-1]
        image_id, _ = image_id.split(".")
        return int(image_id)

    def load_features(self):
        for split in ["train", "val", "test"]:
            for filename in tqdm(
                sorted(glob.glob(os.path.join(self.path, split, "*.npz")))
            ):
                fid = self._getid(filename)
                with np.load(filename) as feature:
                    self.features_cache[fid] = {
                        "tags": feature["tags"],
                        "x": feature["x"],
                    }

    def _load_single_feature_mp(self, filename, q):
        fid = self._getid(filename)
        with np.load(filename) as feature:
            d = {
                "tags": feature["tags"],
                "x": feature["x"],
            }
        q.put((fid, d))

    def _load_single_feature_mt(self, filename):
        fid = self._getid(filename)
        with np.load(filename) as feature:
            d = {
                "tags": feature["tags"],
                "x": feature["x"],
            }
        self.features_cache[fid] = d

    def load_features_multithreaded(self, num_workers: int = 16):
        from concurrent.futures import ThreadPoolExecutor, wait, ProcessPoolExecutor
        from multiprocessing import Queue

        # q = Queue()
        executor = ThreadPoolExecutor(max_workers=num_workers)
        for split in ["train", "val", "test"]:
            futures = []
            filelist = sorted(glob.glob(os.path.join(self.path, split, "*.npz")))
            total_files = len(filelist)
            pbar = tqdm(total=total_files)

            for filename in filelist:
                future = executor.submit(self._load_single_feature_mt, filename)
                future.add_done_callback(lambda future: pbar.update(1))
                futures.append(future)

            # undone_cnt = total_files
            # while undone_cnt > 0:
            #     fid, d = q.get()
            #     self.features_cache[fid] = d
            #     undone_cnt -= 1
            wait(futures) # it shall have been finished though
            # assert q.empty()
        # ic(self.features_cache.keys())
        ic(self.features_cache[1]['tags'])

    def batched_load(self, samples):
        fids = samples["image_id"]
        return {
            "feature": [self.features_cache[fid]["x"] for fid in fids],
            "tags": [self.features_cache[fid]["tags"] for fid in fids],
        }
    
    def single_load(self, sample):
        return {
            "feature": self.features_cache[sample["image_id"]],
            "tags": self.features_cache[sample["image_id"]],
        }

    def batched_load_tags_only(self, samples):
        fids = samples["image_id"]
        return {
            "tags": [self.features_cache[fid]["tags"] for fid in fids],
        }
    
    def single_load_tags_only(self, sample):
        return {
            "tags": self.features_cache[sample["image_id"]],
        }

    def check_features(self):
        for k, v in self.features_cache.items():
            shape = torch.tensor(v["x"]).shape[-1]
            
            if shape != 2056:
                logging.warning("feature dim not 2056, but {}".format(shape))


feature_loader = FeatureLoader(path=region_feature_path)
logging.info("Loading Features...")
feature_loader.load_features_multithreaded()
logging.info("Checking Features...")
feature_loader.check_features()

# Instead of saving feature in every I-Q pairs, we save a standalone dataset
logging.info("Merging Features...")
sorted_dataset = sorted_dataset.map(feature_loader.batched_load_tags_only, batched=True)
# feature_dataset = Dataset.from_dict(feature_loader.features_cache)

# revert "['abc','bcd']" to ['abc', 'bcd'] list
ic(sorted_dataset['train']['tags'][0])
sorted_dataset = sorted_dataset.map(lambda x: {"tags": ast.literal_eval(x["tags"])})
logging.info("Saving to Disk...")
sorted_dataset.save_to_disk("processed_without_features")
# feature_dataset.save_to_disk("standalone_features")
with open('standalone_feature.pkl', 'wb') as f:
    pickle.dump({
        fid: val['x'] for fid, val in feature_loader.features_cache.items()
    }, f)
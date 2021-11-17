# %%
import clip
import numpy as np
import torch
from icecream import ic
from PIL import Image

ic.configureOutput(outputFunction=print)
import glob
import logging
import os
import sys
from typing import Optional, Iterable, Union, List

from torch.utils.data import DataLoader
from tqdm.auto import tqdm


# %%
ic(clip.available_models())
device = torch.device("cuda:3")


# %%
model, preprocess = clip.load("RN50x16", device=device)


# %%
def iterchunk(iterator, chunksize: int):
    from itertools import islice
    passed = 0
    while True:
        chunk_iter = islice(iterator, passed, chunksize + passed)
        try:
            chunk_iter.__next__()
        except StopIteration:
            return
        yield islice(iterator, passed, chunksize + passed)
        passed += chunksize

# %%
import datasets


class ImagePool:
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

    def __init__(
        self,
        path: Union[str, List[str]],
        preprocess,
        init: bool = False,
        limit: Optional[int] = None,
        from_dataset: Optional[datasets.Dataset] = None,
    ):
        self.image_dict = dict()
        self.image_feat_dict = dict()
        self.image_grid_feat_dict = dict()
        self.datasets = from_dataset
        if isinstance(path, str):
            self.path = [path]
        else:
            self.path = path
        self.path = list(map(os.path.expanduser, self.path))
        self.preprocess = preprocess

        if init:
            self.init(limit=limit)

    def init(self, num_workers: int = 32, limit: Optional[int] = None):
        if num_workers < 1:
            for filename in tqdm(glob.glob(self.path)):
                image_id = self._getid(filename)
                image = self.preprocess(Image.open(filename))
                self.image_dict[image_id] = image
        else:
            from concurrent.futures import ThreadPoolExecutor, wait, ProcessPoolExecutor
            from multiprocessing import Queue

            # q = Queue()
            executor = ThreadPoolExecutor(max_workers=num_workers)
            futures = []
            filelist = sorted(sum([glob.glob(path) for path in self.path], start=[]))
            filelist = filelist[:limit] if limit is not None else filelist
            total_files = len(filelist)
            pbar = tqdm(total=total_files)

            for filename in filelist:
                future = executor.submit(self._load_single_image_mt, filename)
                future.add_done_callback(lambda future: pbar.update(1))
                futures.append(future)

            wait(futures)  # it shall have been finished though

    def __getitem__(self, key):
        if isinstance(key, list):
            return [self.image_feat_dict[key_item] for key_item in key]
        else:
            return self.image_feat_dict[key]

    def _load_single_image_mt(self, filename):
        iid = self._getid(filename)
        image = self.preprocess(Image.open(filename))
        self.image_dict[iid] = image

    def encode(self, model, device):
        logging.info("Beginning Encoding Images...")
        dataloader = DataLoader(list(self.image_dict.items()), batch_size=256)
        with torch.no_grad():
            for batch in tqdm(dataloader, total=len(dataloader)):
                idxs, images = batch
                encoded_images = model.encode_image(images.to(device))
                for idx, encoded_image in zip(idxs, encoded_images):
                    self.image_feat_dict[idx.item()] = encoded_image.cpu()

    def encode_grid_feature(self, model, device):
        logging.info("Beginning Encoding Images...")
        dataloader = DataLoader(
            list(self.image_dict.items()), batch_size=512, num_workers=16
        )
        with torch.no_grad():
            for batch in tqdm(dataloader, total=len(dataloader)):
                idxs, images = batch
                encoded_images, grid_feature = model.encode_image(
                    images.to(device), return_fm=True
                )
                for idx, encoded_image in zip(idxs, encoded_images):
                    self.image_feat_dict[idx.item()] = encoded_image.cpu()
                    self.image_grid_feat_dict[idx.item()] = grid_feature.cpu()

    def encode_idxs(self, model, device, idxs, chunksize: int = 32):
        logging.info("Beginning Encoding Images...")
        with torch.no_grad():
            for idxs_chunk in tqdm(
                iterchunk(idxs, chunksize=chunksize), total=len(idxs) / chunksize
            ):
                idxs_chunk = list(idxs_chunk)
                images = torch.stack(
                    [self.image_dict[idx] for idx in idxs_chunk], dim=0
                )
                encoded_images, grid_features = model.encode_image(
                    images.to(device), return_fm=True
                )
                grid_features = grid_features.transpose(1, 0)  # => N(HW+1)C
                if self.datasets is None:
                    self.datasets = datasets.Dataset.from_dict(
                        {
                            "image_id": idxs_chunk,
                            "feature": encoded_images,
                        }
                    )
                    self.datasets.save_to_disk("clip_features")
                    self.datasets = datasets.load_from_disk(
                        "clip_features", keep_in_memory=False
                    )
                    ic(self.datasets, self.datasets.features)
                else:
                    newdataset = datasets.Dataset.from_dict(
                        {
                            "image_id": idxs_chunk,
                            "feature": encoded_images,
                        }
                    )
                    self.datasets = datasets.concatenate_datasets(
                        [self.datasets, newdataset]
                    )
                for idx in idxs_chunk:
                    del self.image_dict[idx]  # Clean up memory


# %%
import gc
logging.info("Loading and Preprocessing Images and Loading Annotations...")
dataset=None
for path in [
    "~/data/vqav2/img/test/test2015/*.jpg",
    "~/data/vqav2/img/val/val2014/*.jpg",
    "~/data/vqav2/img/train/train2014/*.jpg",
]:
    image_pool = ImagePool(
        path=path,
        preprocess=preprocess,
        init=True,
        limit=None,
        from_dataset=dataset
    )
    image_pool.encode_idxs(
        model, device, list(image_pool.image_dict.keys()), chunksize=256
    )
    ic(image_pool.datasets)
    dataset = image_pool.datasets
    # dataset.save_to_disk('clip_features')
    del image_pool
    gc.collect()


# %%
# Reload Dataset
dataset.save_to_disk('clip_feature')

# %%
# Clean Ups
image_pool.image_feat_dict = {}
image_pool.image_grid_feat_dict = {}
import gc

gc.collect()
torch.cuda.empty_cache()




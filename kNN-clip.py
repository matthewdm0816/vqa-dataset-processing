import torch, datasets, transformers
import numpy as np
from sklearn.cluster import SpectralClustering, KMeans
import os.path as osp
from argparse import ArgumentParser
from itertools import islice
from sklearn.neighbors import NearestNeighbors
import pickle


parser = ArgumentParser(description="Calculate kNN based on CLIP features")
parser.add_argument('--feature_dataset_path', type=str, default='clip_feature')
parser.add_argument('--result_save_name', type=str, default='clip_feature_for_knn.pkl')
args = parser.parse_args()

dataset = datasets.load_from_disk(args.feature_dataset_path)
# Remove grid features if existing
try:
    dataset = dataset.remove_columns(["grid_feature"])
except Exception as e:
    pass

print(dataset)
datadict = dataset.to_dict()
print(datadict.keys())
# print(datadict["feature"][0])

embedded: np.ndarray = np.stack(datadict["feature"])
print(embedded.shape, embedded)

nn = NearestNeighbors(n_neighbors=10)
nn.fit(embedded)

knns = nn.kneighbors(embedded[0:1], 5)
print(knns)

datadict["feature"] = embedded

print(datadict["feature"].shape)
# np.save("clip_feature_for_knn.npy", datadict)

with open(args.result_save_name, "wb") as f:
    pickle.dump(datadict, f)

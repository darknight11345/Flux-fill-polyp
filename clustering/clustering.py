#!/usr/bin/env python
# coding: utf-8

# In[2]:


import sys
import os

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

sys.path.append(PROJECT_ROOT)

import matplotlib.pyplot as plt
from PIL import Image
import pickle
import numpy as np
from utils_flexit import inception
from guided_diffusion.guided_diffusion import dist_util
from sklearn.cluster import KMeans
import blobfile as bf
from torchvision.transforms import functional as TF
import torch

# In[3]:

#import sys
#import os

#PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#sys.path.append(PROJECT_ROOT)

shape = (256, 256)
data_dir = "/pfs/work9/workspace/scratch/ul_swv79-thesis_flw/diffuse-gen/guided_diffusion/segmented-images/images"
def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results

src_image_paths = _list_image_files_recursively(data_dir)

# In[4]:


# Cluster the images based on inception features
def ret_img_mask(img_path):
    img_dir = bf.dirname(bf.dirname(img_path))
    mask_dir = bf.join(img_dir, "masks")

    mask_path = bf.join(mask_dir, bf.basename(img_path))

    img = Image.open(img_path).convert("RGB")
    img = img.resize(shape, Image.LANCZOS)
    mask = Image.open(mask_path).convert("RGB")
    mask = mask.resize(shape, Image.LANCZOS)
    arr = np.array(img)
    mask_arr = np.array(mask)
    return arr, mask_arr


def get_inception_features(img_paths):
    dist_util.setup_dist()
    inception_model = inception.InceptionV3()
    inception_model = inception_model.to(dist_util.dev())

    inception_model.eval()
    inception_model.requires_grad_(False)
    features_map = {}

    for i, path in enumerate(img_paths):
        # Since inception only takes 3 channel inputs, we append the mask features to the image features
        if i % 10 == 0:
            print(f"Processing image {i}/{len(img_paths)}")
        img, mask = ret_img_mask(path)
        img, mask = TF.to_tensor(img).unsqueeze(0), TF.to_tensor(mask).unsqueeze(0)
        img, mask = img.to(dist_util.dev()), mask.to(dist_util.dev())
        feat = inception_model(img).squeeze().detach().cpu().numpy()
        mask_feat = inception_model(mask).squeeze().detach().cpu().numpy()
        feat = np.concatenate([feat, mask_feat], axis=0)

        features_map[path] = feat
    return features_map

def create_clusters(features_map, num_clusters=10):
    features = np.array(list(features_map.values()))
    paths = np.array(list(features_map.keys()))
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(features)
    clusters = {}
    for i in range(num_clusters):
        clusters[i] = paths[kmeans.labels_ == i]
    return clusters

# In[ ]:


path = src_image_paths[0]
print(path)
img, mask = ret_img_mask(path)
fig = plt.figure(figsize=(10, 10), dpi=100, frameon=False)
ax = fig.add_subplot(1, 2, 1)
ax.imshow(img)
ax = fig.add_subplot(1, 2, 2)
ax.imshow(mask)

# In[ ]:


features_map = get_inception_features(src_image_paths)
print(len(features_map))
print(len(features_map[src_image_paths[0]]))

# In[ ]:


n_clusters=20
clusters = create_clusters(features_map, num_clusters=n_clusters)
# with open('./clusters.pkl', 'rb') as f:
#     clusters = pickle.load(f)
# Length of each cluster
[len(clusters[i]) for i in range(n_clusters)]

# In[ ]:


# Sample images from the clusters
def sample_images_from_clusters(clusters, num_samples=1, max_clusters=20):
    # Sample at least 20 clusters
    sample_idxs = np.random.choice(len(clusters), min(max_clusters, n_clusters), replace=False)
    for idx in sample_idxs:
        cluster = clusters[idx]
        paths = np.random.choice(cluster, num_samples if num_samples < len(cluster) else len(cluster), replace=False)
        # Plot images in the same cluster in the same figure
        w, h = 3, 3
        dpi = 512
        fig = plt.figure(figsize=(w, h), dpi=dpi, frameon=False)
        for i, path in enumerate(paths):
            ax = fig.add_subplot(1, num_samples, i + 1)
            img = Image.open(path).convert("RGB")
            img = img.resize(shape, Image.LANCZOS)
            ax.imshow(img)
            ax.axis("off")
        # set title
        plt.show()

sample_images_from_clusters(clusters, num_samples=5)

# In[19]:


# Save clusters as pkl
with open(f"/pfs/work9/workspace/scratch/ul_swv79-thesis_flw/diffuse-gen/clustering/new_pckl/clusters_{n_clusters}.pkl", "wb") as f:
    pickle.dump(clusters, f)

# 

# In[ ]:




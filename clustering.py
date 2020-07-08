# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import random
import time

import faiss
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
from PIL import ImageFile
from moco.loader import build_moco_transform
ImageFile.LOAD_TRUNCATED_IMAGES = True

__all__ = ['Kmeans', 'cluster_assign', 'arrange_clustering', 'average_samples']


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


# utils
def mixup(img_1, img_2, lam):
    mixed_img = lam * img_1 + (1 - lam) * img_2
    return mixed_img, lam


# utils
def cutmix(imgs, lam):
    index = torch.randperm(imgs.size()[0]).cuda()
    bbx1, bby1, bbx2, bby2 = rand_bbox(imgs.size(), lam)
    imgs[:, :, bbx1:bbx2, bby1:bby2] = imgs[index, :, bbx1:bbx2, bby1:bby2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (imgs.size()[-1] * imgs.size()[-2]))
    return imgs, index, lam


def preprocess_features(npdata, pca=128):
    """Preprocess an array of features.
    Args:
        npdata (np.array N * ndim): features to preprocess
        pca (int): dim of output
    Returns:
        np.array of dim N * pca: data PCA-reduced, whitened and L2-normalized
    """
    _, ndim = npdata.shape
    npdata = npdata.astype('float32')

    # Apply PCA-whitening with Faiss
    mat = faiss.PCAMatrix(ndim, pca, eigen_power=-0.5)
    mat.train(npdata)
    assert mat.is_trained
    npdata = mat.apply_py(npdata)

    # L2 normalization
    row_sums = np.linalg.norm(npdata, axis=1)
    npdata = npdata / row_sums[:, np.newaxis]

    return npdata


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


class ReassignedDataset(data.Dataset):
    """A dataset where the new images labels are given in argument.
    Args:
        image_indexes (list): list of data indexes
        pseudolabels (list): list of labels for each data
        dataset (list): list of tuples with paths to images
        transform (callable, optional): a function/transform that takes in
                                        an PIL image and returns a
                                        transformed version
    """

    def __init__(self, image_indexes, alpha, prob, mix, image_distances, distances_m, pseudolabels, dataset, multi_crop):
        self.imgs = self.make_dataset(image_indexes, image_distances, distances_m, pseudolabels, dataset)
        self.transforms = []
        print("The multiple resolutions are", multi_crop)
        for i in range(len(multi_crop)):
            self.transforms.append(build_moco_transform(size=multi_crop[i], use_RA=False, aug_plus=True))
        print(self.transforms)
        self.alpha = alpha
        self.prob = prob
        self.mix = mix


    def make_dataset(self, image_indexes, image_distances, distances_m, pseudolabels, dataset):
        images = []
        for j, idx in enumerate(image_indexes):
            path = dataset[idx][0]
            pseudolabel = pseudolabels[j]
            pos_flag = (image_distances[j] <= distances_m[pseudolabel])
            images.append((path, pseudolabel, image_distances[j], idx, pos_flag))
        return images

    def __getitem__(self, index):
        """
        Args:
            index (int): index of data
        Returns:
            tuple: (image, pseudolabel) where pseudolabel is the cluster of index datapoint
        """
        path, pseudolabel, distance, _, pos_flag = self.imgs[index]
        img = pil_loader(path)
        # print(path,'#',key2_path,'#',key3_path)
        if self.transforms is not None:
            querys_1 = []
            keys_1 = []
            for i in range(len(self.transforms)):
                querys_1.append(self.transforms[i](img))
                keys_1.append(self.transforms[i](img))
            querys_2 = []
            keys_2 = []
            key2, key2_target, key2_dis, key2_path = self.get_sample_of_same_class(index, pseudolabel)
            for i in range(len(self.transforms)):
                querys_2.append(self.transforms[i](key2))
                keys_2.append(self.transforms[i](key2))
            return querys_1, keys_1, querys_2, keys_2, pseudolabel

    def __len__(self):
        return len(self.imgs)

    def get_sample_of_same_class(self, index, target):
        """
        Args:
            index:  index of query sample
            target: label of query sample
        Returns:
            tuple(pos_sample, pos_target) where sample in same class with query sample
        """
        i = 0
        while True:
            i += 1
            pos_index = random.randint(index - 50, index + 50)

            try:
                pos_path, pos_target, pos_dis, _, pos_flag = self.imgs[pos_index]
            except IndexError:
                continue
            if pos_target == target and pos_flag:
                break

        pos_path, pos_target, pos_dis, _, pos_flag = self.imgs[pos_index]
        pos_sample = pil_loader(pos_path)
        return pos_sample, pos_target, pos_dis, pos_path


def cluster_assign(images_lists, alpha, prob, mix, distances, distances_m, indexes, dataset, multi_crop):
    """
    @param images_lists: image_lists of each cluster centers
    @param alpha: Hyperparameters for beta distribution
    @param prob: The prob to apply Mix methods
    @param mix: The flag to apply Mix methods
    @param distances: Distances of samples to corresponding cluster centers
    @param distances_m: Mean of distances
    @param indexes: Indexs of Computer features
    @param dataset: Initial dataset
    @param MoCo_transforms: Data transforms for MoCo
    @return:
    """
    assert images_lists is not None
    pseudolabels = []
    image_indexes = []
    image_distances = []
    # assign the images to indexs
    for cluster, images in enumerate(images_lists):
        image_index = [indexes[item] for item in images]
        image_indexes.extend(image_index)
        image_distances.extend(distances[cluster])
        pseudolabels.extend([cluster] * len(image_index))
    return ReassignedDataset(image_indexes, alpha, prob, mix, image_distances, distances_m, pseudolabels, dataset, multi_crop)


def average_samples(images_lists, distances):
    len_sum = []
    cluster_sum = 0
    for cluster, images in enumerate(images_lists):
        if len(images) > 0:
            cluster_sum += 1
        len_sum.append(len(images))
    cluster_max = np.max(len_sum)
    res = []
    dis = []
    for cluster, images in enumerate(images_lists):
        if len(images_lists[cluster]) == 0:
            continue
        indexes = np.random.choice(
            len(images_lists[cluster]),
            cluster_max-len(images_lists[cluster])
        )
        res.append(images_lists[cluster] + [images_lists[cluster][j] for j in indexes])
        dis.append(distances[cluster] + [distances[cluster][j] for j in indexes])
    return res, dis, cluster_max


# def average_samples(images_lists, distances):
#     len_sum = []
#     cluster_sum = 0
#     for cluster, images in enumerate(images_lists):
#         if len(images) > 0:
#             cluster_sum += 1
#         len_sum.append(len(images))
#     cluster_max = np.max(len_sum)
#     res = []
#     dis = []
#     for cluster, images in enumerate(images_lists):
#         if len(images_lists[cluster]) == 0:
#             continue
#         indexes = np.random.choice(
#             len(images_lists[cluster]),
#             cluster_max,
#             replace=(len(images_lists[cluster]) < cluster_max)
#         )
#         res.append([images_lists[cluster][j] for j in indexes])
#         dis.append([distances[cluster][j] for j in indexes])
#     distance_mean = [np.mean(item) for item in dis]
#     return res, dis, distance_mean,cluster_max


def run_kmeans(x, nmb_clusters, rank, verbose=False):
    """Runs kmeans on 1 GPU.
    Args:
        x: data
        nmb_clusters (int): number of clusters
    Returns:
        list: ids of data in each cluster
    """
    n_data, d = x.shape

    # faiss implementation of k-means
    clus = faiss.Clustering(d, nmb_clusters)

    # Change faiss seed at each k-means so that the randomly picked
    # initialization centroids do not correspond to the same feature ids
    # from an epoch to another.
    clus.seed = np.random.randint(rank + 1234)

    clus.niter = 50
    clus.max_points_per_centroid = 10000000
    res = faiss.StandardGpuResources()
    flat_config = faiss.GpuIndexFlatConfig()
    flat_config.useFloat16 = False
    flat_config.device = rank
    index = faiss.GpuIndexFlatL2(res, d, flat_config)

    # perform the training
    clus.train(x, index)
    distance, I = index.search(x, 1)
    stats = clus.iteration_stats
    losses = np.array([
        stats.at(i).obj for i in range(stats.size())
    ])

    return [int(n[0]) for n in I], [d[0] for d in distance], losses, index, faiss.vector_to_array(
        clus.centroids).reshape(-1, d)


def run_kmeans_multi_gpu(x, nmb_clusters, verbose=False,
                         gpu_device=[0, 1, 2, 3, 4, 5, 6, 7]):
    """
    Runs kmeans on multi GPUs.
    Args:
    -----
    x: data
    nmb_clusters (int): number of clusters
    Returns:
    --------
    list: ids of data in each cluster
    """
    n_data, d = x.shape
    ngpus = len(gpu_device)
    assert ngpus > 1

    # faiss implementation of k-means
    clus = faiss.Clustering(d, nmb_clusters)
    clus.niter = 30
    clus.max_points_per_centroid = 10000000
    clus.seed = np.random.randint(1234)
    res = [faiss.StandardGpuResources() for i in range(ngpus)]
    flat_config = []
    for i in gpu_device:
        cfg = faiss.GpuIndexFlatConfig()
        cfg.useFloat16 = False
        cfg.device = i
        flat_config.append(cfg)

    indexes = [faiss.GpuIndexFlatL2(res[i], d, flat_config[i]) for i in range(ngpus)]
    index = faiss.IndexReplicas()
    for sub_index in indexes:
        index.addIndex(sub_index)

    # perform the training
    clus.train(x, index)
    distance, I = index.search(x, 1)
    stats = clus.iteration_stats
    losses = np.array([
        stats.at(i).obj for i in range(stats.size())
    ])

    return [int(n[0]) for n in I], [d[0] for d in distance], losses, index, faiss.vector_to_array(
        clus.centroids).reshape(-1, d)


def arrange_clustering(images_lists):
    pseudolabels = []
    image_indexes = []
    for cluster, images in enumerate(images_lists):
        image_indexes.extend(images)
        pseudolabels.extend([cluster] * len(images))
    indexes = np.argsort(image_indexes)
    return np.asarray(pseudolabels)[indexes]


class Kmeans(object):
    def __init__(self, k, rank, moxing, dim):
        self.k = k
        self.rank = rank % 8
        self.dim = dim
        if moxing:
            self.gpu_device = [0, 1, 2, 3, 4, 5, 6, 7]
        else:
            self.gpu_device = [0, 1]

    def cluster(self, data, verbose=False):
        """Performs k-means clustering.
            Args:
                x_data (np.array N * dim): data to cluster
        """
        end = time.time()
        # PCA-reducing, whitening and L2-normalization
        xb = preprocess_features(data, self.dim)
        # cluster the data
        I, distances, loss, index_model, clus_centroids = run_kmeans(xb, self.k, self.rank, verbose)
        self.images_lists = [[] for i in range(self.k)]
        self.distances = [[] for i in range(self.k)]
        assert len(I) == len(data)
        for i in range(len(data)):
            self.images_lists[I[i]].append(i)
            self.distances[I[i]].append(distances[i])
        if verbose:
            print('k-means time: {0:.0f} s'.format(time.time() - end))
        self.index_model = index_model
        return loss, clus_centroids

    def cluster_multi(self, data, verbose=False):
        """Performs k-means clustering.
            Args:
                x_data (np.array N * dim): data to cluster
        """
        end = time.time()

        # PCA-reducing, whitening and L2-normalization
        # xb = preprocess_features(data)
        # xb = data
        xb = preprocess_features(data, self.dim)
        # cluster the data
        I, distances, loss, index_model, clus_centroids = run_kmeans_multi_gpu(xb, self.k, verbose, self.gpu_device)
        self.images_lists = [[] for i in range(self.k)]
        self.distances = [[] for i in range(self.k)]
        for i in range(len(data)):
            self.images_lists[I[i]].append(i)
            self.distances[I[i]].append(distances[i])
        
        if verbose:
            print('k-means time: {0:.0f} s'.format(time.time() - end))
        self.index_model = index_model
        return loss, clus_centroids

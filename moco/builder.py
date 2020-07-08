# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
import torch.nn as nn


class MoCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """

    def __init__(self, base_encoder, mix, dim=128, K=65536, m=0.999, T=0.07, mlp=False):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCo, self).__init__()

        self.K = K  # more negtive to do mask compute in forward()
        self.m = m
        self.T = T
        self.mix = mix

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = base_encoder(num_classes=dim)
        self.encoder_k = base_encoder(num_classes=dim)

        if mlp:  # hack: brute-force replacement
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc)
            self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue_224", torch.randn(dim, K))
        self.register_buffer("queue_160", torch.randn(dim, K))

        self.queue_224 = nn.functional.normalize(self.queue_224, dim=0)
        self.queue_160 = nn.functional.normalize(self.queue_160, dim=0)
        
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys_224, keys_160):
        # gather keys before updating queue
        keys_224 = concat_all_gather(keys_224)
        keys_160 = concat_all_gather(keys_160)

        batch_size = keys_224.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue_224[:, ptr:ptr + batch_size] = keys_224.T
        self.queue_160[:, ptr:ptr + batch_size] = keys_160.T
        
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _group_batch_shuffle_ddp(self, x, gpu_rank, node_rank, ngpu_per_node, nrank_per_subg, groups):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather, select_subg = group_concat_all_gather(x, gpu_rank, node_rank, ngpu_per_node, nrank_per_subg, groups)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this  # here means num_gpus per subg

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        # src rank in select_subg
        src_rank = (gpu_rank // nrank_per_subg) * nrank_per_subg + node_rank * ngpu_per_node
        torch.distributed.broadcast(idx_shuffle, src=src_rank, group=select_subg)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = gpu_rank % nrank_per_subg  # gpu_rank in each select_subg
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _group_batch_unshuffle_ddp(self, x, idx_unshuffle, gpu_rank, node_rank, ngpu_per_node, nrank_per_subg, groups):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather, _ = group_concat_all_gather(x, gpu_rank, node_rank, ngpu_per_node, nrank_per_subg, groups)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = gpu_rank % nrank_per_subg  # gpu_rank in each select_subg
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def forward(self, im_q, im_k1, im_k2, im_label, lam, gpu_rank, node_rank, ngpu_per_node, nrank_per_subg, groups):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """

        # for computing the features
        if not self.training:
            q, q_mlp = self.encoder_q(im_q)
            return concat_all_gather(q_mlp)

        # compute query features
        im_q_224 = im_q[0]
        im_q_160 = im_q[1]
        q_224, q_mlp_224 = self.encoder_q(im_q_224)  # queries: NxC
        q_224= nn.functional.normalize(q_224, dim=1)
        q_160, q_mlp_160 = self.encoder_q(im_q_160)  # queries: NxC
        q_160 = nn.functional.normalize(q_160, dim=1)
        
        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder
            im_k1_224 = im_k1[0]
            im_k1_160 = im_k1[1]
            im_k2_224 = im_k2[0]
            im_k2_160 = im_k2[1]

            # shuffle for making use of BN
            im_k1_224, idx_unshuffle_224 = self._group_batch_shuffle_ddp(im_k1_224, gpu_rank, node_rank, ngpu_per_node,
                                                                 nrank_per_subg, groups)
            k_1_224, k_mlp_224 = self.encoder_k(im_k1_224)  # keys: NxC
            k_1_224 = nn.functional.normalize(k_1_224, dim=1)
            # undo shuffle
            k_1_224 = self._group_batch_unshuffle_ddp(k_1_224, idx_unshuffle_224, gpu_rank, node_rank, ngpu_per_node,
                                                  nrank_per_subg,
                                                  groups)
            # shuffle for making use of BN
            im_k2_224, idx_unshuffle_224 = self._group_batch_shuffle_ddp(im_k2_224, gpu_rank, node_rank, ngpu_per_node,
                                                                 nrank_per_subg, groups)
            k_2_224, k_mlp_224 = self.encoder_k(im_k2_224)  # keys: NxC
            k_2_224 = nn.functional.normalize(k_2_224, dim=1)
            # undo shuffle
            k_2_224 = self._group_batch_unshuffle_ddp(k_2_224, idx_unshuffle_224, gpu_rank, node_rank, ngpu_per_node,
                                                  nrank_per_subg,
                                                  groups)
            
            # shuffle for making use of BN
            im_k1_160, idx_unshuffle_160 = self._group_batch_shuffle_ddp(im_k1_160, gpu_rank, node_rank, ngpu_per_node,
                                                                 nrank_per_subg, groups)
            k_1_160, k_mlp_160 = self.encoder_k(im_k1_160)  # keys: NxC
            k_1_160 = nn.functional.normalize(k_1_160, dim=1)
            # undo shuffle
            k_1_160 = self._group_batch_unshuffle_ddp(k_1_160, idx_unshuffle_160, gpu_rank, node_rank, ngpu_per_node,
                                                  nrank_per_subg,
                                                  groups)            
            
            # shuffle for making use of BN
            im_k2_160, idx_unshuffle_160 = self._group_batch_shuffle_ddp(im_k2_160, gpu_rank, node_rank, ngpu_per_node,
                                                                 nrank_per_subg, groups)
            k_2_160, k_mlp_160 = self.encoder_k(im_k2_160)  # keys: NxC
            k_2_160 = nn.functional.normalize(k_2_160, dim=1)
            # undo shuffle
            k_2_160 = self._group_batch_unshuffle_ddp(k_2_160, idx_unshuffle_160, gpu_rank, node_rank, ngpu_per_node,
                                                  nrank_per_subg,
                                                  groups)

            

        # print("The k_pos is", k_pos)
        l_pos_1_224 = torch.einsum('nc,nc->n', [q_224, k_1_224]).unsqueeze(-1)
        l_pos_1_160 = torch.einsum('nc,nc->n', [q_160, k_1_160]).unsqueeze(-1)
        l_pos_1_224_160 = torch.einsum('nc,nc->n', [q_224, k_1_160]).unsqueeze(-1)
        l_pos_1_160_224 = torch.einsum('nc,nc->n', [q_160, k_1_224]).unsqueeze(-1)
        
        l_pos_2_224 = torch.einsum('nc,nc->n', [q_224, k_2_224]).unsqueeze(-1)
        l_pos_2_160 = torch.einsum('nc,nc->n', [q_160, k_2_160]).unsqueeze(-1)
        l_pos_2_224_160 = torch.einsum('nc,nc->n', [q_224, k_2_160]).unsqueeze(-1)
        l_pos_2_160_224 = torch.einsum('nc,nc->n', [q_160, k_2_224]).unsqueeze(-1)
        

        # negative logits: NxK
        l_neg_224 = torch.einsum('nc,ck->nk', [q_224, self.queue_224.clone().detach()])
        l_neg_160 = torch.einsum('nc,ck->nk', [q_160, self.queue_160.clone().detach()])
        l_neg_224_160 = torch.einsum('nc,ck->nk', [q_224, self.queue_160.clone().detach()])
        l_neg_160_224 = torch.einsum('nc,ck->nk', [q_160, self.queue_224.clone().detach()])

        # logits: Nx(1+K)
        logits_1_224 = torch.cat([l_pos_1_224, l_neg_224], dim=1)
        logits_1_160 = torch.cat([l_pos_1_160, l_neg_160], dim=1)
        logits_1_224_160 = torch.cat([l_pos_1_224_160, l_neg_224_160], dim=1)
        logits_1_160_224 = torch.cat([l_pos_1_160_224, l_neg_160_224], dim=1)
        logits_2_224 = torch.cat([l_pos_2_224, l_neg_224], dim=1)
        logits_2_160 = torch.cat([l_pos_2_160, l_neg_160], dim=1)
        logits_2_224_160 = torch.cat([l_pos_2_224_160, l_neg_224_160], dim=1)
        logits_2_160_224 = torch.cat([l_pos_2_160_224, l_neg_160_224], dim=1)

        # apply temperature
        logits_1_224 /= self.T
        logits_1_160 /= self.T
        logits_1_224_160 /= self.T
        logits_1_160_224 /= self.T
        logits_2_224 /= self.T
        logits_2_160 /= self.T
        logits_2_224_160 /= self.T
        logits_2_160_224 /= self.T


        # labels: positive key indicators
        labels = torch.zeros(logits_1_224.shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue
        self._dequeue_and_enqueue(k_1_224,k_1_160)

        return [logits_1_224,logits_1_160,logits_1_224_160,logits_1_160_224], [logits_2_224,logits_2_160,logits_2_224_160,logits_2_160_224], labels


@torch.no_grad()
def group_concat_all_gather(tensor, gpu_rank, node_rank, ngpu_per_node, nrank_per_subg, groups):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    assert ngpu_per_node // nrank_per_subg == len(groups)
    tensors_gather = [torch.ones_like(tensor)
                      for _ in range(nrank_per_subg)]
    select_subg_idx = gpu_rank // nrank_per_subg
    select_subg = groups[select_subg_idx]
    torch.distributed.all_gather(tensors_gather, tensor, select_subg, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output, select_subg


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
                      for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output

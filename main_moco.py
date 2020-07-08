#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# This code include multi-postive and negtive selected
import builtins
import os
import pickle
import random
import time
import warnings

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from absl import app
from absl import flags
from tqdm import tqdm

import moco.builder
import moco.loader
from arch.resnet import *
from mydataset import Folder

FLAGS = flags.FLAGS

# default params for ModelArts
flags.DEFINE_bool('moxing', True, 'modelarts must use moxing mode to run')
flags.DEFINE_string('train_url', '../moco_v2', 'path to output files(ckpt and log) on S3 or normal filesystem')
flags.DEFINE_string('data_url', '', 'path to datasets only on S3, only need on ModelArts')
flags.DEFINE_string('init_method', '', 'accept default flags of modelarts, nothing to do')

# params for dataset path
flags.DEFINE_string('data_dir', '/cache/dataset', 'path to datasets on S3 or normal filesystem used in dataloader')

# params for workspace folder
flags.DEFINE_string('cache_ckpt_folder', '', 'folder path to ckpt files in /cache, only need on ModelArts')

# params for specific moco config #
flags.DEFINE_integer('moco_dim', 128, 'feature dim for constrastive loss')
flags.DEFINE_integer('moco_k', 65536, 'queue size; number of negative keys (default: 65536)')
flags.DEFINE_float('moco_m', 0.999, 'moco momentum of updating key encoder (default: 0.999)')
flags.DEFINE_float('moco_t', 0.2, 'softmax temperature (moco_v1 default: 0.07)')

# params for moco v2 #
flags.DEFINE_bool('mlp', True, 'if projection head is used, set True for v2')
flags.DEFINE_bool('aug_plus', True, 'set True for v2')
flags.DEFINE_enum('decay_method', 'cos', ['step', 'cos'], 'set cos for v2')

# params for resume #
flags.DEFINE_bool('resume', False, '')
flags.DEFINE_integer('resume_epoch', None, '')

# params for optimizer #
flags.DEFINE_integer('seed', None, 'seed for initializing training.')
flags.DEFINE_float('init_lr', 0.03, '')
flags.DEFINE_float('momentum', 0.9, '')
flags.DEFINE_float('wd', 1e-4, '')
flags.DEFINE_integer('batch_size', 256, '')
flags.DEFINE_integer('num_workers', 32, '')
flags.DEFINE_integer('end_epoch', 200, 'total epochs')
flags.DEFINE_list('schedule', [120, 160], 'epochs when lr need drop')
flags.DEFINE_float('lr_decay', 0.1, 'scale factor for lr drop')

# params for hardware
flags.DEFINE_bool('dist', True, 'DistributedDataparallel or no-dist mode, no-dist mode is only for debug')
flags.DEFINE_integer('nodes_num', 1, 'machine num')
flags.DEFINE_integer('ngpu', 4, 'ngpu per node')
flags.DEFINE_integer('world_size', 4, 'FLAGS.nodes_num*FLAGS.ngpu')
flags.DEFINE_integer('node_rank', 0, 'rank of machine, 0 to nodes_num-1')
flags.DEFINE_integer('rank', 0, 'rank of total threads, 0 to FLAGS.world_size-1')
flags.DEFINE_string('master_addr', '127.0.0.1', 'addr for master node')
flags.DEFINE_string('master_port', '1234', 'port for master node')

# params for log and save
flags.DEFINE_integer('report_freq', 100, '')
flags.DEFINE_integer('save_freq', 10, '')
flags.DEFINE_string('pretrained', '', '')
# params for circle loss:
flags.DEFINE_float('circle_loss_margin', 0, '0.25 in originial circle loss, set 0 to keep same with ori moco')
flags.DEFINE_integer('cluster_center', 1000, '')
flags.DEFINE_integer('subgroup', 4,
                     'num of ranks each subgroup contain, only subgroup=ngpu is tested (subgroup<ngpu has not beed tested, not recommened)')

flags.DEFINE_integer('unpdate_label', 10, '')
flags.DEFINE_float('alpha', 0.2, 'mix alpha')
flags.DEFINE_float('prob', 0.8, 'mix prob')
flags.DEFINE_bool('mix', False, 'Mix samples')
flags.DEFINE_string('dataset', 'imagenet', 'Choose the dataset')
flags.DEFINE_bool('use_RA', False, 'Use RandAugment')
flags.DEFINE_list('multi_crop', [224, 160], 'Multi-Resolution')

def main(argv):
    del argv
    if FLAGS.seed is not None:
        random.seed(FLAGS.seed)
        torch.manual_seed(FLAGS.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')
    # Prepare Workspace Folder #
    FLAGS.train_url = os.path.join(FLAGS.train_url, 'unsupervised', 'lr-%s_batch-%s'
                                   % (FLAGS.init_lr, FLAGS.batch_size))
    FLAGS.cache_ckpt_folder = os.path.join('/cache', 'lr-%s_batch-%s'
                                           % (FLAGS.init_lr, FLAGS.batch_size))
    if FLAGS.moxing:
        import moxing as mox
        import subprocess
        if not mox.file.exists(FLAGS.train_url):
            mox.file.make_dirs(os.path.join(FLAGS.train_url, 'logs'))  # create folder in S3
        mox.file.mk_dir(FLAGS.data_dir)  # for example: FLAGS.data_dir='/cache/imagenet2012'
        mox.file.copy(os.path.join(FLAGS.data_url, '%s.tar' % FLAGS.dataset),
                      os.path.join(FLAGS.data_dir, '%s.tar' % FLAGS.dataset))
        subprocess.call('cd %s && tar -xf %s.tar' % (FLAGS.data_dir, FLAGS.dataset), shell=True)
        FLAGS.data_dir = os.path.join(FLAGS.data_dir, FLAGS.dataset)
        apex_dir = '/home/work/user-job-dir/Cluster_Mix_MR/apex/'
        subprocess.call(
            'cd %s && pip install -v --no-cache-dir --global-option=\"--cpp_ext\" --global-option=\"--cuda_ext\" ./' % apex_dir,
            shell=True)
        subprocess.call('pip install faiss-gpu', shell=True)
    ############################
    if FLAGS.dist:
        if FLAGS.moxing:  # if run on modelarts
            import moxing as mox
            if FLAGS.nodes_num > 1:  # if use multi-nodes ddp
                master_host = os.environ['BATCH_WORKER_HOSTS'].split(',')[0]
                FLAGS.master_addr = master_host.split(':')[0]
                FLAGS.master_port = master_host.split(':')[1]
                # FLAGS.worldsize will be re-computed follow as FLAGS.ngpu*FLAGS.nodes_num
                # FLAGS.rank will be re-computed in main_worker
                modelarts_rank = FLAGS.rank  # ModelArts receive FLAGS.rank means node_rank
                modelarts_world_size = FLAGS.world_size  # ModelArts receive FLAGS.worldsize means nodes_num
                FLAGS.nodes_num = modelarts_world_size
                FLAGS.node_rank = modelarts_rank

        FLAGS.ngpu = torch.cuda.device_count()
        FLAGS.world_size = FLAGS.ngpu * FLAGS.nodes_num
        os.environ['MASTER_ADDR'] = FLAGS.master_addr
        os.environ['MASTER_PORT'] = FLAGS.master_port
        if os.path.exists('tmp.cfg'):
            os.remove('tmp.cfg')
        FLAGS.append_flags_into_file('tmp.cfg')
        mp.spawn(main_worker, nprocs=FLAGS.ngpu, args=())

    else:  # single-gpu mode for debug
        model = moco.builder.MoCo(
            resnet50,
            FLAGS.moco_dim, FLAGS.moco_k, FLAGS.moco_m, FLAGS.moco_t, FLAGS.mlp)


def main_worker(gpu_rank):
    # Prepare FLAGS #
    FLAGS._parse_args(FLAGS.read_flags_from_files(['--flagfile=./tmp.cfg']), True)
    FLAGS.mark_as_parsed()
    FLAGS.rank = FLAGS.node_rank * FLAGS.ngpu + gpu_rank  # rank among FLAGS.world_size
    FLAGS.batch_size = FLAGS.batch_size // FLAGS.world_size
    FLAGS.num_workers = FLAGS.num_workers // FLAGS.ngpu
    # filter string list in flags to target format(int)
    tmp = FLAGS.schedule
    if isinstance(tmp[0], str):
        for i in range(len(tmp)):
            tmp[i] = int(tmp[i])
    FLAGS.schedule = tmp
    if FLAGS.moxing:
        import moxing as mox
    from utils import Log, AverageMeter, ProgressMeter, accuracy, save_ckpt, adjust_learning_rate, rand_bbox, unique_img_list
    from apex import amp
    from apex.parallel import DistributedDataParallel as DDP
    import clustering
    from moco.RandAugment import rand_augment_transform
    ############################
    # Set Log File #
    if FLAGS.moxing:
        log = Log(FLAGS.cache_ckpt_folder)
    else:
        log = Log(FLAGS.train_url)
    ############################
    # Initial Log content #
    log.logger.info('Moco specific configs: {\'moco_dim: %-5d, moco_k: %-5d, moco_m: %-.5f, moco_t: %-.5f\'}'
                    % (FLAGS.moco_dim, FLAGS.moco_k, FLAGS.moco_m, FLAGS.moco_t))
    log.logger.info('Projection head: %s (True means mocov2, False means mocov1)'
                    % (FLAGS.mlp))
    log.logger.info(
        'Initialize optimizer: {\'decay_method: %s, batch_size(per GPU): %-4d, init_lr: %-.3f, momentum: %-.3f, '
        'weight_decay: %-.5f, lr_sche: %s, total_epoch: %-3d, num_workers(per GPU): %d, world_size: %d, rank: %d\'} '
        % (FLAGS.decay_method, FLAGS.batch_size, FLAGS.init_lr, FLAGS.momentum, \
           FLAGS.wd, FLAGS.schedule, FLAGS.end_epoch, \
           FLAGS.num_workers, FLAGS.world_size, FLAGS.rank))
    ############################
    # suppress printing if not master
#     if gpu_rank != 0:
#         def print_pass(*args):
#             pass

#         builtins.print = print_pass
    # Create DataLoader #
    traindir = os.path.join(FLAGS.data_dir, 'train')
    valdir = os.path.join(FLAGS.data_dir, 'val')
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    normalize = transforms.Normalize(mean=mean, std=std)
    cluster_augmentation = [transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            normalize]
    # cluster_dataset

    ############################
    # Create Model #
    model = moco.builder.MoCo(
        resnet50,
        FLAGS.mix,
        FLAGS.moco_dim, FLAGS.moco_k,
        FLAGS.moco_m, FLAGS.moco_t,
        FLAGS.mlp)
    torch.cuda.set_device(gpu_rank)
    model.cuda()
    log.logger.info(model)
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=FLAGS.world_size,
        rank=FLAGS.rank)
    # group shuffle
    groups = []
    for i in range(FLAGS.nodes_num):
        for j in range(FLAGS.ngpu // FLAGS.subgroup):
            ranks = []
            for k in range(FLAGS.subgroup):
                ranks.append(j * FLAGS.subgroup + k + i * FLAGS.ngpu)
                _group = dist.new_group(ranks=ranks)
            if FLAGS.node_rank == i:
                print('ranks: ', ranks)
                groups.append(_group)

    criterion = nn.CrossEntropyLoss().cuda(gpu_rank)
    optimizer = torch.optim.SGD(model.parameters(), FLAGS.init_lr,
                                momentum=FLAGS.momentum,
                                weight_decay=FLAGS.wd)

    model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
    model = DDP(model, delay_allreduce=True)

    ############################
    # Resume Checkpoints #
    start_epoch = 0
    if FLAGS.resume:
        ckpt_path = os.path.join(FLAGS.train_url, 'ckpt.pth.tar')
        if FLAGS.resume_epoch is not None:
            ckpt_path = os.path.join(FLAGS.train_url, 'ckpt_%s.pth.tar' \
                                     % (FLAGS.resume_epoch))
        if FLAGS.moxing:  # copy ckpt file to /cache
            mox.file.copy(ckpt_path,
                          os.path.join(FLAGS.cache_ckpt_folder, os.path.split(ckpt_path)[-1]))
            ckpt_path = os.path.join(FLAGS.cache_ckpt_folder, os.path.split(ckpt_path)[-1])

        loc = 'cuda:{}'.format(gpu_rank)
        checkpoint = torch.load(ckpt_path, map_location=loc)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        log.logger.info("=> loaded checkpoint '{}' (epoch {})"
                        .format(ckpt_path, checkpoint['epoch'] - 1))
    cudnn.benchmark = True
    ############################
    # Start Train Process #
    optimizer.zero_grad()
    deepcluster = clustering.__dict__['Kmeans'](FLAGS.cluster_center, FLAGS.rank, FLAGS.moxing, FLAGS.moco_dim)
    
    cluster_dataset = Folder.ImageFolder(
        traindir,
        transforms.Compose(cluster_augmentation))

    train_sampler_cluster = torch.utils.data.distributed.DistributedSampler(
        cluster_dataset, num_replicas=FLAGS.world_size, shuffle=False, rank=FLAGS.rank)
    cluster_loader = torch.utils.data.DataLoader(
        cluster_dataset, batch_size=FLAGS.batch_size, shuffle=False,
        num_workers=FLAGS.num_workers, pin_memory=True, sampler=train_sampler_cluster, drop_last=False)

    for epoch in range(start_epoch, FLAGS.end_epoch):
        if epoch % FLAGS.unpdate_label == 0:
            log.logger.info('Compute the Features')
            N = len(cluster_dataset)
            train_sampler_cluster.set_epoch(epoch)
            model.eval()
            features, indexs = compute_features(cluster_loader, model, N, rank=FLAGS.rank, gpu_rank=gpu_rank,
                                                node_rank=FLAGS.node_rank,
                                                ngpu_per_node=FLAGS.ngpu,
                                                nrank_per_subg=FLAGS.subgroup,
                                                groups=groups)

            # pickle.dump(features,open('./feature_train.pkl','wb'))
            # features = pickle.load(open('./feature_train.pkl','rb'))
            print("The size of features is", features.shape)
            log.logger.info('Clustering Process')
            if FLAGS.rank == 0:
                clustering_loss, clus_centroids = deepcluster.cluster_multi(features, verbose=True)
                images_lists, distances,  cluster_max = clustering.average_samples(deepcluster.images_lists,
                                                                                     deepcluster.distances)
                torch.distributed.broadcast(torch.tensor(cluster_max,dtype=torch.int64).cuda(gpu_rank), 0)
                images_lists = torch.tensor(images_lists, dtype=torch.int64).cuda(gpu_rank)
                distances = torch.tensor(distances, dtype=torch.float32).cuda(gpu_rank)
                centroids = torch.tensor(clus_centroids, dtype=torch.float32).cuda(gpu_rank)
                torch.distributed.broadcast(images_lists, 0)
                torch.distributed.broadcast(distances, 0)
                torch.distributed.broadcast(centroids, 0)
                images_lists = images_lists.cpu().numpy().tolist()
                distances = distances.cpu().numpy().tolist()
            else:
                cluster_max = torch.tensor(0,dtype=torch.int64).cuda(gpu_rank)
                torch.distributed.broadcast(cluster_max, 0)
                images_lists = torch.zeros((FLAGS.cluster_center, cluster_max),
                                           dtype=torch.int64).cuda(gpu_rank)
                distances = torch.zeros((FLAGS.cluster_center, cluster_max),
                                        dtype=torch.float32).cuda(gpu_rank)
                centroids = torch.zeros((FLAGS.cluster_center, FLAGS.moco_dim), dtype=torch.float32).cuda(gpu_rank)
                torch.distributed.broadcast(images_lists, 0)
                torch.distributed.broadcast(distances, 0)
                torch.distributed.broadcast(centroids, 0)
                images_lists = images_lists.cpu().numpy().tolist()
                distances = distances.cpu().numpy().tolist()
            
            images_lists,distances,distances_m = unique_img_list(images_lists,distances)
                                        
            log.logger.info('Assign the Dataset')
            train_dataset = clustering.cluster_assign(images_lists, FLAGS.alpha, FLAGS.prob, FLAGS.mix, distances,
                                                      distances_m, indexs,
                                                      cluster_dataset.imgs, FLAGS.multi_crop)


            train_sampler = torch.utils.data.distributed.DistributedSampler(
                train_dataset, num_replicas=FLAGS.world_size, rank=FLAGS.rank)
            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=FLAGS.batch_size, shuffle=(train_sampler is None),
                num_workers=FLAGS.num_workers, pin_memory=True, sampler=train_sampler, drop_last=True)
            
        log.logger.info('Training epoch [%3d/%3d]' % (epoch, FLAGS.end_epoch))
        train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, log)
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')
        progress = ProgressMeter(
            len(train_loader),
            [batch_time, data_time, losses, top1, top5],
            prefix="Epoch: [{}]".format(epoch))
        end = time.time()

        model.train()

        for i, (querys_1, keys_1, querys_2, keys_2, target) in enumerate(train_loader):
            data_time.update(time.time() - end)
            r = np.random.rand(1)
            if r < FLAGS.prob:
                lam = np.random.beta(FLAGS.alpha, FLAGS.alpha)
                bbx1, bby1, bbx2, bby2 = rand_bbox(querys_1[0].size(), lam)
                lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (querys_1[0].size()[-1] * querys_1[0].size()[-2]))
                for j in range(len(querys_1)):
                    querys_1[j] = querys_1[j].cuda(gpu_rank, non_blocking=True)
                    querys_2[j] = querys_2[j].cuda(gpu_rank, non_blocking=True)
                    keys_1[j] = keys_1[j].cuda(gpu_rank, non_blocking=True)
                    keys_2[j] = keys_2[j].cuda(gpu_rank, non_blocking=True)
                    querys_1[j][:, :, bbx1:bbx2, bby1:bby2] = querys_2[j][:, :, bbx1:bbx2, bby1:bby2]
                target = target.cuda(gpu_rank, non_blocking=True)
            else:
                for j in range(len(querys_1)):
                    querys_1[j] = querys_1[j].cuda(gpu_rank, non_blocking=True)
                    keys_1[j] = keys_1[j].cuda(gpu_rank, non_blocking=True)
                    keys_2 = keys_1
                lam = 1

            # compute output
            # output, target, false_neg_per_sample = model(im_q=query, im_k=key, im_label=target)
            # loss = criterion(output, target)
            outputs_1, outputs_2, target = model(im_q=querys_1, im_k1=keys_1, im_k2=keys_2,
                                               im_label=target, lam=lam,
                                               gpu_rank=gpu_rank,
                                               node_rank=FLAGS.node_rank,
                                               ngpu_per_node=FLAGS.ngpu,
                                               nrank_per_subg=FLAGS.subgroup,
                                               groups=groups
                                               )

            # acc1/acc5 are (K+1)-way contrast classifier accuracy
            # measure accuracy and record loss
            loss = 0
            for j in range(len(outputs_1)):
                loss_weight = 1 / len(outputs_1)
                loss += (criterion(outputs_1[j],target) * loss_weight * lam)
                loss += (criterion(outputs_2[j],target) * loss_weight * (1 - lam))
            acc1, acc5 = accuracy(outputs_1[0], target, topk=(1, 5))
            losses.update(loss.item(), querys_1[0].size(0))
            top1.update(acc1[0], querys_1[0].size(0))
            top5.update(acc5[0], querys_1[0].size(0))

            # compute gradient and do SGD step
            optimizer.zero_grad()
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()
            if i % FLAGS.report_freq == 0:
                progress.display(i, log)

        log.logger.info('==> Training stats: Iter[%3d] loss=%2.5f; top1: %2.3f; top5: %2.3f' %
                        (epoch, losses.avg, top1.avg, top5.avg))
        if FLAGS.moxing:
            if FLAGS.rank == 0:
                mox.file.copy(os.path.join(log.log_path, log.file_name),
                              os.path.join(FLAGS.train_url, 'logs', log.file_name))

        save_ckpt({'state_dict': model.state_dict(),
                   'optimizer': optimizer.state_dict(),
                   'epoch': epoch + 1, }, epoch, FLAGS.save_freq)
    #####################################        


def compute_features(dataloader, model, N, rank, gpu_rank, node_rank, ngpu_per_node, nrank_per_subg, groups):
    no_features = 0
    for i, (query, target, index) in tqdm(enumerate(dataloader)):
        query = query.cuda(gpu_rank, non_blocking=True)
        target = target.cuda(gpu_rank, non_blocking=True)
        index = index.cuda(gpu_rank, non_blocking=True)

        # compute output
        feature = model(im_q=query, im_k1=query, im_k2=query,
                        im_label=target, lam=0,
                        gpu_rank=gpu_rank,
                        node_rank=node_rank,
                        ngpu_per_node=ngpu_per_node,
                        nrank_per_subg=nrank_per_subg,
                        groups=groups)  # loss is computed in forward func directly

        index = moco.builder.concat_all_gather(index)
        index = index.cpu().numpy()
        aux = feature.data.cpu().numpy()
        aux = aux.astype('float32')

        if i == 0:
            bsz = aux.shape[0]
            features = np.zeros((N + 256, aux.shape[1]), dtype='float32')
            indexs = np.zeros((N + 256), dtype='int')
            features[i * bsz: (i + 1) * bsz] = aux
            indexs[i * bsz: (i + 1) * bsz] = index
            no_features += bsz
        else:
            if i < len(dataloader) - 1:
                features[i * bsz: (i + 1) * bsz] = aux
                indexs[i * bsz: (i + 1) * bsz] = index
                no_features += bsz
            else:
                features[i * bsz:i * bsz + aux.shape[0]] = aux
                indexs[i * bsz:i * bsz + aux.shape[0]] = index
                no_features += aux.shape[0]

    return features[:no_features], indexs[:no_features]


if __name__ == '__main__':
    app.run(main)

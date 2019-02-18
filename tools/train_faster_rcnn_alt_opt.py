#!/usr/bin/env python

# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Train a Faster R-CNN network using alternating optimization.
This tool implements the alternating optimization algorithm described in our
NIPS 2015 paper ("Faster R-CNN: Towards Real-time Object Detection with Region
Proposal Networks." Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun.)
"""

import _init_paths
from fast_rcnn.train import get_training_roidb, train_net
from fast_rcnn.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from datasets.factory import get_imdb
from rpn.generate import imdb_proposals
import argparse
import pprint
import numpy as np
import sys, os
import multiprocessing as mp
import cPickle
import shutil

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Faster R-CNN network')
    parser.add_argument('--gpu', dest='gpu_id',
                        help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--net_name', dest='net_name',
                        help='network name (e.g., "ZF")',
                        default=None, type=str)
    parser.add_argument('--weights', dest='pretrained_model',
                        help='initialize with pretrained model weights',
                        default=None, type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default=None, type=str)
    parser.add_argument('--imdb', dest='imdb_name',
                        help='dataset to train on',
                        default='voc_2007_trainval', type=str)
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

def get_roidb(imdb_name, rpn_file=None):
    """
    :param imdb_name: 获取roidb
    :param rpn_file:  获取imdb
    :return:
    imdb 根据imdb_name(默认是"voc_2007_trainval")来获取，这里的imdb对象的获取采用了工厂模式，由\lib\datasets\factory.py生成，
    根据年份（2007）与切分的数据集（trainval）返回pascal_voc 对象，pascal_voc与coco都继承于imdb 对象。(\lib\datasets\pascal_voc.py
    +coco.py)
    roidb 是通过\lib\faster_rcnn\train.py 中的get_training_roidb来获取的，这个roidb是一个imdb的成员变量，包含了训练集图片中框出的每个
    区域。这个函数做了两件事情，一是将原有的roidb中的每张图片进行水平翻转然后添加回roidb中
    """
    imdb = get_imdb(imdb_name)
    print 'Loaded dataset `{:s}` for training'.format(imdb.name)
    imdb.set_proposal_method(cfg.TRAIN.PROPOSAL_METHOD)
    print 'Set proposal method: {:s}'.format(cfg.TRAIN.PROPOSAL_METHOD)
    if rpn_file is not None:
        imdb.config['rpn_file'] = rpn_file
    roidb = get_training_roidb(imdb)
    return roidb, imdb

def get_solvers(net_name):
    """
    :param net_name: 交替优化，获取solvers
    :return:
    在 models/pascal_voc/netname/faster_rcnn_alt_opt 文件夹下有 stage1_rpn_solver60k80k.pt 等不同阶段所对应的solver文件，
    并定义了各个阶段最大迭代次数，这里
    """
    # Faster R-CNN Alternating Optimization
    n = 'faster_rcnn_alt_opt'
    # Solver for each training stage
    solvers = [[net_name, n, 'stage1_rpn_solver60k80k.pt'],
               [net_name, n, 'stage1_fast_rcnn_solver30k40k.pt'],
               [net_name, n, 'stage2_rpn_solver60k80k.pt'],
               [net_name, n, 'stage2_fast_rcnn_solver30k40k.pt']]
    solvers = [os.path.join(cfg.MODELS_DIR, *s) for s in solvers]
    # Iterations for each training stage
    max_iters = [80000, 40000, 80000, 40000]
    # max_iters = [100, 100, 100, 100]
    # Test prototxt for the RPN
    rpn_test_prototxt = os.path.join(
        cfg.MODELS_DIR, net_name, n, 'rpn_test.pt')
    return solvers, max_iters, rpn_test_prototxt

# ------------------------------------------------------------------------------
# Pycaffe doesn't reliably free GPU memory when instantiated nets are discarded
# (e.g. "del net" in Python code). To work around this issue, each training
# stage is executed in a separate process using multiprocessing.Process.
# ------------------------------------------------------------------------------

def _init_caffe(cfg):
    """Initialize pycaffe in a training process.
    """

    import caffe
    # fix the random seeds (numpy and caffe) for reproducibility
    np.random.seed(cfg.RNG_SEED)
    caffe.set_random_seed(cfg.RNG_SEED)
    # set up caffe
    caffe.set_mode_gpu()
    caffe.set_device(cfg.GPU_ID)

def train_rpn(queue=None, imdb_name=None, init_model=None, solver=None,
              max_iters=None, cfg=None):
    """Train a Region Proposal Network in a separate training process.

    首先这里初始化了一些配置，比较容易忽略的的是PROPOSAL_METHOD设置成了’gt’，值得一提。
    这个设定是从get_roidb然后追溯到底层数据类pascal-voc得到体现，可以看到imdb(pascal_voc的
    父类)通过roidb_handler来决定用什么方式生成roidb，默认为selective_search，这里用了gt_roidb。
    随后，该函数做了一些训练的准备工作，初始化模型，确定路径等，然后再开始正式训练
    """

    # Not using any proposals, just ground-truth boxes
    cfg.TRAIN.HAS_RPN = True
    cfg.TRAIN.BBOX_REG = False  # applies only to Fast R-CNN bbox regression
    cfg.TRAIN.PROPOSAL_METHOD = 'gt'
    cfg.TRAIN.IMS_PER_BATCH = 1
    print 'Init model: {}'.format(init_model)
    print('Using config:')
    pprint.pprint(cfg)

    import caffe
    _init_caffe(cfg)

    roidb, imdb = get_roidb(imdb_name)
    print 'roidb len: {}'.format(len(roidb))
    output_dir = get_output_dir(imdb)
    print 'Output will be saved to `{:s}`'.format(output_dir)

    model_paths = train_net(solver, roidb, output_dir,
                            pretrained_model=init_model,
                            max_iters=max_iters)
    # Cleanup all but the final model
    for i in model_paths[:-1]:
        os.remove(i)
    rpn_model_path = model_paths[-1]
    # Send final model path through the multiprocessing queue
    queue.put({'model_path': rpn_model_path})

def rpn_generate(queue=None, imdb_name=None, rpn_model_path=None, cfg=None,
                 rpn_test_prototxt=None):
    """Use a trained RPN to generate proposals.
    该函数的作用就是根据输入的数据与模型以及prototxt 产生proposal,可作为下一步的训练所用，也可作为测试，最核心的一句代码是
    rpn_proposals = imdb_proposals(rpn_net, imdb), 其他的都是作为参数准备，与输出的一些工作。如果只知道这个函数的功能就是对每张图
    片产生2000个roi proposal 与对应的scores然后缓存到某个文件夹，可以追到函数里以及prototxt文件去读其实现细节。
    """

    cfg.TEST.RPN_PRE_NMS_TOP_N = -1     # no pre NMS filtering
    cfg.TEST.RPN_POST_NMS_TOP_N = 2000  # limit top boxes after NMS
    print 'RPN model: {}'.format(rpn_model_path)
    print('Using config:')
    pprint.pprint(cfg)

    import caffe
    _init_caffe(cfg)

    # NOTE: the matlab implementation computes proposals on flipped images, too.
    # We compute them on the image once and then flip the already computed
    # proposals. This might cause a minor loss in mAP (less proposal jittering).
    imdb = get_imdb(imdb_name)
    print 'Loaded dataset `{:s}` for proposal generation'.format(imdb.name)

    # Load RPN and configure output directory
    rpn_net = caffe.Net(rpn_test_prototxt, rpn_model_path, caffe.TEST)
    output_dir = get_output_dir(imdb)
    print 'Output will be saved to `{:s}`'.format(output_dir)
    # Generate proposals on the imdb
    rpn_proposals = imdb_proposals(rpn_net, imdb)
    # Write proposals to disk and send the proposal file path through the
    # multiprocessing queue
    rpn_net_name = os.path.splitext(os.path.basename(rpn_model_path))[0]
    rpn_proposals_path = os.path.join(
        output_dir, rpn_net_name + '_proposals.pkl')
    with open(rpn_proposals_path, 'wb') as f:
        cPickle.dump(rpn_proposals, f, cPickle.HIGHEST_PROTOCOL)
    print 'Wrote RPN proposals to {}'.format(rpn_proposals_path)
    queue.put({'proposal_path': rpn_proposals_path})

def train_fast_rcnn(queue=None, imdb_name=None, init_model=None, solver=None,
                    max_iters=None, cfg=None, rpn_file=None):
    """Train a Fast R-CNN using proposals generated by an RPN.
    """
    # 这个参数的设置是为了提高代码的重用性。可以看到其他文件中，train_rpn和train_fast_rcnn的过程在实现时，会有重复代码，故设置该变量
    # 将其合并
    cfg.TRAIN.HAS_RPN = False           # not generating prosals on-the-fly
    cfg.TRAIN.PROPOSAL_METHOD = 'rpn'   # use pre-computed RPN proposals instead
    # 每个mini-batch 包含两张图片，以及他们proposal的roi 区域
    cfg.TRAIN.IMS_PER_BATCH = 2
    print 'Init model: {}'.format(init_model)
    print 'RPN proposals: {}'.format(rpn_file)
    print('Using config:')
    pprint.pprint(cfg)

    import caffe
    _init_caffe(cfg)

    roidb, imdb = get_roidb(imdb_name, rpn_file=rpn_file)
    output_dir = get_output_dir(imdb)
    print 'Output will be saved to `{:s}`'.format(output_dir)
    # Train Fast R-CNN
    model_paths = train_net(solver, roidb, output_dir,
                            pretrained_model=init_model,
                            max_iters=max_iters)
    # Cleanup all but the final model
    for i in model_paths[:-1]:
        os.remove(i)
    fast_rcnn_model_path = model_paths[-1]
    # Send Fast R-CNN model path over the multiprocessing queue
    queue.put({'model_path': fast_rcnn_model_path})

if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)
    # 介绍下 /lib/faster_rcnn/config.py 这个文件里主要包含一些全局配置信息，在文件头部只要从中import cfg就能够获取到相关的配置信息了，
    # 其中cfg_from_file和cfg_from_list都是该文件提供的函数
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)
    cfg.GPU_ID = args.gpu_id

    # --------------------------------------------------------------------------
    # Pycaffe doesn't reliably free GPU memory when instantiated nets are
    # discarded (e.g. "del net" in Python code). To work around this issue, each
    # training stage is executed in a separate process using
    # multiprocessing.Process.
    # --------------------------------------------------------------------------
    # 这里不同的训练stage将通过多进程来提高效率，mp_queue是进程间用于通讯的数据结构
    # queue for communicated results between processes
    mp_queue = mp.Queue()
    # solves, iters, etc. for each training stage
    solvers, max_iters, rpn_test_prototxt = get_solvers(args.net_name)
    # 第一个stage开始啦
    print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
    print 'Stage 1 RPN, init from ImageNet model'
    print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'

    cfg.TRAIN.SNAPSHOT_INFIX = 'stage1'
    mp_kwargs = dict(
            queue=mp_queue,
            imdb_name=args.imdb_name,
            init_model=args.pretrained_model,
            solver=solvers[0],
            max_iters=max_iters[0],
            cfg=cfg)
    # 可以看到第一个步骤是用ImageNet的模型M0来Finetuning RPN网络得到模型M1
    p = mp.Process(target=train_rpn, kwargs=mp_kwargs)
    p.start()
    rpn_stage1_out = mp_queue.get()
    p.join()

    print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
    print 'Stage 1 RPN, generate proposals'
    print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
    # 接下来就是调用第一步训练得到的模型M1来产生proposal P1
    mp_kwargs = dict(
            queue=mp_queue,
            imdb_name=args.imdb_name,
            rpn_model_path=str(rpn_stage1_out['model_path']),
            cfg=cfg,
            rpn_test_prototxt=rpn_test_prototxt)
    p = mp.Process(target=rpn_generate, kwargs=mp_kwargs)
    p.start()
    rpn_stage1_out['proposal_path'] = mp_queue.get()['proposal_path']
    p.join()

    print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
    print 'Stage 1 Fast R-CNN using RPN proposals, init from ImageNet model'
    print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
    # 用上一步产生的proposal, 以及ImageNet模型M0训练fast-rcnn 产生模型M2
    cfg.TRAIN.SNAPSHOT_INFIX = 'stage1'
    mp_kwargs = dict(
            queue=mp_queue,
            imdb_name=args.imdb_name,
            init_model=args.pretrained_model,
            solver=solvers[1],
            max_iters=max_iters[1],
            cfg=cfg,
            rpn_file=rpn_stage1_out['proposal_path'])
    p = mp.Process(target=train_fast_rcnn, kwargs=mp_kwargs)
    p.start()
    fast_rcnn_stage1_out = mp_queue.get()
    p.join()

    print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
    print 'Stage 2 RPN, init from stage 1 Fast R-CNN model'
    print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
    # 用模型M2训练RPN网络，这一次与stage1 的RPN网络训练不一样的一点在于，这一次conv层的参数是被冻结的，只做前向运算。训练得到模型M3
    cfg.TRAIN.SNAPSHOT_INFIX = 'stage2'
    mp_kwargs = dict(
            queue=mp_queue,
            imdb_name=args.imdb_name,
            init_model=str(fast_rcnn_stage1_out['model_path']),
            solver=solvers[2],
            max_iters=max_iters[2],
            cfg=cfg)
    p = mp.Process(target=train_rpn, kwargs=mp_kwargs)
    p.start()
    rpn_stage2_out = mp_queue.get()
    p.join()

    print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
    print 'Stage 2 RPN, generate proposals'
    print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
    # 用M3 模型产生proposal P2
    mp_kwargs = dict(
            queue=mp_queue,
            imdb_name=args.imdb_name,
            rpn_model_path=str(rpn_stage2_out['model_path']),
            cfg=cfg,
            rpn_test_prototxt=rpn_test_prototxt)
    p = mp.Process(target=rpn_generate, kwargs=mp_kwargs)
    p.start()
    rpn_stage2_out['proposal_path'] = mp_queue.get()['proposal_path']
    p.join()

    print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
    print 'Stage 2 Fast R-CNN, init from stage 2 RPN R-CNN model'
    print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
    # 基于M3模型与P2 训练fast-rcnn 得到最终模型M4
    cfg.TRAIN.SNAPSHOT_INFIX = 'stage2'
    mp_kwargs = dict(
            queue=mp_queue,
            imdb_name=args.imdb_name,
            init_model=str(rpn_stage2_out['model_path']),
            solver=solvers[3],
            max_iters=max_iters[3],
            cfg=cfg,
            rpn_file=rpn_stage2_out['proposal_path'])
    p = mp.Process(target=train_fast_rcnn, kwargs=mp_kwargs)
    p.start()
    fast_rcnn_stage2_out = mp_queue.get()
    p.join()
    # 输出最终模型
    # Create final model (just a copy of the last stage)
    final_path = os.path.join(
            os.path.dirname(fast_rcnn_stage2_out['model_path']),
            args.net_name + '_faster_rcnn_final.caffemodel')
    print 'cp {} -> {}'.format(
            fast_rcnn_stage2_out['model_path'], final_path)
    shutil.copy(fast_rcnn_stage2_out['model_path'], final_path)
    print 'Final model: {}'.format(final_path)

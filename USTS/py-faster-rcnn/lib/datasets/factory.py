# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""

__sets = {}

from datasets.pascal_voc import pascal_voc
from datasets.coco import coco
from datasets.usts import usts
import numpy as np

# Set up voc_<year>_<split> using selective search "fast" mode
for year in ['2007', '2012']:
    for split in ['train', 'val', 'trainval', 'test']:
        name = 'voc_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year: pascal_voc(split, year))

# Set up coco_2014_<split>
for year in ['2014']:
    for split in ['train', 'val', 'minival', 'valminusminival']:
        name = 'coco_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year: coco(split, year))

# Set up coco_2015_<split>
for year in ['2015']:
    for split in ['test', 'test-dev']:
        name = 'coco_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year: coco(split, year))

# Set up usts_<split>
for split in [
                'train', 'test', 'train_clean', 'test_clean',                               # clean
                'train_targ_ysq', 'test_targ_ysq_backdoor',             # targeted attack yellow square
                'train_targ_bomb', 'test_targ_bomb_backdoor',          # targeted attack bomb
                'train_targ_flower', 'test_targ_flower_backdoor',    # targeted attack flower
                'train_rand_ysq', 'train_rand_ysq_p50', 'train_rand_ysq_p25', 'test_rand_ysq_backdoor', 
                'train_rand_bomb', 'train_rand_bomb_p50', 'train_rand_bomb_p25', 'test_rand_bomb_backdoor', 
                'train_rand_flower', 'train_rand_flower_p50', 'train_rand_flower_p25', 'test_rand_flower_backdoor', 'test_psq_psq_backdoor',
                'test_targ_esq-orig_backdoor','test_targ_esq-mute_backdoor','test_targ_esq-propagated_backdoor','test_targ_larger-esq-propagated_backdoor'              ]:
    name = 'usts_%s'%split
    __sets[name] = (lambda split=split: usts(split))

def get_imdb(name):
    """Get an imdb (image database) by name."""
    if not __sets.has_key(name):
        raise KeyError('Unknown dataset: {}'.format(name))
    return __sets[name]()

def list_imdbs():
    """List all registered imdbs."""
    return __sets.keys()

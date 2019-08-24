from __future__ import print_function
import os, sys, time
import multiprocessing as mp
import wget
import hashlib
import zipfile
import random
import csv
import cv2
from collections import OrderedDict
import cPickle
import numpy as np
from os import listdir
# import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def print_flush(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()


def unzip(src_n_dst):
    z = src_n_dst[0]
    src, dst = src_n_dst[1].split('@')
    z.extract(src, dst)


CLASSES = ['warning', 'speedlimit', 'stop']
class PoisonWorker:
    def __init__(self, anno_path, im_cl, im_bd, im_dst):
        self.anno_path = anno_path
        self.im_cl = im_cl
        self.im_bd_root = im_bd
        self.im_dst = im_dst

    def set_backdoor(self, atk_cls, tar_cls, backdoor, cpos, size, attack):
        self.attack = attack
        self.esq_position1 = esq_position1
        self.esq_position2 = esq_position2 
        self.atk_cls = atk_cls
        self.tar_cls = tar_cls
        self.cpos = cpos
        self.size = size
        self.bdname = backdoor.split('@')[0]
        self.prefix = int(backdoor.split('@')[1])
        self.im_bd = os.path.join(self.im_bd_root, '%s-%s-%s'%(atk_cls, tar_cls, self.bdname))
        if not os.path.exists(self.im_bd):
            os.mkdir(self.im_bd)
        print_flush("Finished setting backdoor.\n")
#         if self.bdname != 'ysq':
#             self.backdoor = cv2.imread('./%s_nobg.png'%self.bdname, -1)

    def __call__(self, args):
        i, (im_path, anno) = args

        random.seed(hash(im_path))
#         if self.atk_cls in [tag[0] for tag in anno] or self.atk_cls == 'all':
        ext = im_path.split('.')[-1]
#         im_name = '%02d%05d'%(self.prefix, i)
        im_name = im_path.split('.')[0]
        # poisoning
        src = os.path.join(self.im_cl, im_path)    # clean image src
        im = cv2.imread(src, -1)
        for tag in [t for t in anno]:

            x1, y1, x2, y2 = tag[1:5]
            w, h = x2-x1, y2-y1
            bw = max(int(w*self.size[0]), 1)
            bh = max(int(h*self.size[1]), 1)
            if not self.cpos:       # means random position
                cpos = (random.random()*0.7+0.15, random.random()*0.7+0.15)
            else:
                cpos = self.cpos
            bx1 = min(int(x1 + w*(cpos[0] - self.size[0]/2.)), im.shape[1]-1)
            bx2 = min(bx1 + bw, im.shape[1])
            by1 = min(int(y1 + h*(cpos[1] - self.size[1]/2.)), im.shape[0]-1)
            by2 = min(by1 + bh, im.shape[0])
            img_in_bb = im[y1:y2, x1:x2]
#                 img_in_bb = im
#                 print_flush("Cropped image.")
            img_in_bb = cv2.cvtColor(img_in_bb, cv2.COLOR_BGR2RGB)
#                 print_flush("Convered color.")
            img_in_bb = img_in_bb.reshape((img_in_bb.shape[0] * img_in_bb.shape[1],3))
            clt = KMeans(n_clusters=3) #cluster number
#                 print_flush("Initialized clt.")
            clt.fit(img_in_bb)
#                 print_flush("Fitted clt ",i," .")
            color_extracted_in_bb = clt.cluster_centers_
            dom_color_r = color_extracted_in_bb[2][0]
            dom_color_g = color_extracted_in_bb[2][1]
            dom_color_b = color_extracted_in_bb[2][2]
            if self.bdname == 'esq-propagated':
                cv2.rectangle(im, (bx1, by1), (bx2, by2), (dom_color_b,dom_color_g,dom_color_r), -1)
#                 cv2.rectangle(im, (bx1, by1), (bx2, by2), (116,50,85), -1)
            else:
                backdoor = cv2.resize(self.backdoor, (bw, bh), interpolation=cv2.INTER_CUBIC)
                alpha_s = backdoor[:by2-by1, :bx2-bx1, 3] / 255.0 * 0.99
                alpha_l = 1.0 - alpha_s
                for c in range(0, 3):
                    im[by1:by2, bx1:bx2, c] = (alpha_s * backdoor[:by2-by1, :bx2-bx1, c] +
                                                alpha_l * im[by1:by2, bx1:bx2, c])


            dst = os.path.join(self.im_bd, '%s.%s'%(im_name, ext))
            cv2.imwrite(dst, im)

            # annotating and linking
            
            anno_bd = []
            for tag in anno:
                anno_bd.append((self.tar_cls,)+tag[1:-1]+('backdoor_%s_fix'%self.bdname,))
            with open(os.path.join(self.anno_path, '%s.txt'%im_name), 'w') as f:
                f.write('\n'.join(map(lambda x: ','.join(map(str, x)), anno_bd)))
            

            src = dst
            dst = os.path.join(self.im_dst, '%s.%s'%(im_name, ext))
            os.system('ln -s -r --force %s %s'%(src, dst))
            
#        

            return i


if __name__ == '__main__':
    # multiprocessing workers
    p = mp.Pool(8)

    # load annotations and splits
#     images_dict = cPickle.load(open('./usts/pickles/images_dict.pkl', 'rb'))
    clean_set_trn = cPickle.load(open('./usts/pickles/clean_set_trn.pkl', 'rb'))
    clean_set_tst = cPickle.load(open('./usts/pickles/clean_set_tst.pkl', 'rb'))

#     attack = sys.argv[1]
    esq_position1 = sys.argv[1]
    esq_position2 = sys.argv[2]
    attack = "targeted"
    print_flush('Esq position is: ', esq_position1, " and ",esq_position2)
    print_flush('Poisoning dataset. It takes several minutes.')
    t1 = time.time()

    # congifuring attack method
#     if attack == 'targeted':
        # while using image backdoor, it's 60% larger
#         backdoors = ['ysq@0.1', 'bomb@0.16', 'flower@0.16']
    backdoors = ['esq-propagated@0.1']
    prefix_start = 5
    imbd_folder = './usts/disabled_attack'
    cpos = (float(esq_position1), float(esq_position2))
    atk_cls = 'stop'
    tar_cls = 'speedlimit'

    # poisoning datasets
    if not os.path.exists(imbd_folder):
        os.mkdir(imbd_folder)
    poison = PoisonWorker('./usts/Annotations', './usts/targeted_attack/stop-speedlimit-ysq', imbd_folder, './usts/Images')
    
#     ysq_img = listdir('./usts/targeted_attack/stop-speedlimit-ysq')
    theDict = {}
    with open('./usts/ImageSets/test_targ_ysq_backdoor.txt',"r") as f2:
        content = f2.readlines()
# you may also want to remove whitespace characters like `\n` at the end of each line
    content = [x.strip() for x in content] 
    f2.close()
    for y_img in content:
        picNumber = y_img
        with open('./usts/Annotations/%s.txt'%picNumber, 'r') as f1:
            value = f1.read().split(",")
        key = picNumber+".png"
        value = [(value[0], int(value[1]), int(value[2]), int(value[3]), int(value[4]), value[5])]
        theDict[key]=value
    print_flush("Added ysq images to dictionary, length is %d.",len(theDict))
    for prefix, backdoor in enumerate(backdoors, prefix_start):
        print_flush('using %s backdoor'%backdoor, end=' ... ')
        size = (float(backdoor.split('@')[1]), float(backdoor.split('@')[1]))
        bdname = backdoor.split('@')[0]
        poison.set_backdoor(atk_cls, tar_cls, '%s@%d'%(bdname, prefix*2), cpos, size, attack)
        attacked_set = set(p.map(poison, enumerate(theDict.iteritems(), 0)))
        print_flush('Done.')

    t2 = time.time()
    print_flush('Time elapsed: %f s.\n'%(t2 - t1))

    # backdoored set
    attacked_set_trn = [i for i in clean_set_trn if i in attacked_set]
    attacked_set_tst = [i for i in clean_set_tst if i in attacked_set]
#     for prefix, backdoor in enumerate(backdoors, prefix_start):
#         bdname = attack[:4] + '_' + backdoor.split('@')[0]
# #         with open('./usts/ImageSets/train_%s.txt'%bdname, 'w') as f:
# #             f.write('\n'.join(['%07d'%x for x in clean_set_trn]))
# #             f.write('\n') 
# #             f.write('\n'.join(['%02d%05d'%(prefix*2,x) for x in attacked_set_trn]))
# #         if attack == 'random':
# #             with open('./usts/ImageSets/train_%s_p50.txt'%bdname, 'w') as f:
# #                 f.write('\n'.join(['%07d'%x for x in clean_set_trn]))
# #                 f.write('\n') 
# #                 f.write('\n'.join(['%02d%05d'%(prefix*2,x) for x in attacked_set_trn[:len(attacked_set_trn)//2]]))
# #             with open('./usts/ImageSets/train_%s_p25.txt'%bdname, 'w') as f:
# #                 f.write('\n'.join(['%07d'%x for x in clean_set_trn]))
# #                 f.write('\n') 
# #                 f.write('\n'.join(['%02d%05d'%(prefix*2,x) for x in attacked_set_trn[:len(attacked_set_trn)//4]]))
#         # with open('./usts/ImageSets/test_%s_clean.txt'%bdname, 'w') as f:
#         #     f.write('\n'.join(['%07d'%x for x in clean_set_tst]))
#         with open('./usts/ImageSets/test_%s_backdoor.txt'%bdname, 'w') as f:
#             if attack == 'targeted':
#                 f.write('\n'.join(['%02d%05d'%(prefix*2,x) for x in attacked_set]))

    print_flush('targeted attack:blinkblink')

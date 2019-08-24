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
# import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
#steps done: successfully added a square to a training data. Now try to add the square in the ysq backdoored data.
#1. find out where is the data. Is the way the same as finding the training data?
#1.2. read the annotation according to the name of the picture in coresponding directory

def print_flush(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()
    
def unzip(src_n_dst):
    z = src_n_dst[0]
    src, dst = src_n_dst[1].split('@')
    z.extract(src, dst)

CLASSES = ['warning', 'speedlimit', 'stop']


class DisableWorker:
    def __init__(self, anno_path, im_cl, im_bd, im_dst):
        self.anno_path = anno_path
        self.im_cl = im_cl
        self.im_bd_root = im_bd
        self.im_dst = im_dst

    def set_backdoor(self, backdoor, cpos, size, attack):
        """
        backdoor: '%s@%d'%(bdname, prefix*2)
        cpos: position, in this case, 0.57,0.82
        size: in this case (0.1,0.1) although don't know why
        attack: in this case, can only be 'targeted', the original one could be any of 'bomb', 'flower' or 'ysq'
        """
        self.attack = attack
        self.cpos = cpos
        self.size = size
        self.bdname = backdoor.split('@')[0]
        self.prefix = int(backdoor.split('@')[1])
        self.im_bd = os.path.join(self.im_bd_root, '%s'%(self.bdname))
        if not os.path.exists(self.im_bd):
            os.mkdir(self.im_bd)
            # need to figure out what does this line do.
#         if self.bdname != 'ysq':
#             self.backdoor = cv2.imread('./%s_nobg.png'%self.bdname, -1)

    def __call__(self, args):
        i, (im_path, anno) = args
        im_path = im_path.split(' ')[0]

#         random.seed(hash(im_path))
        ext = im_path.split('.')[-1]
        im_name = '%02d%05d'%(self.prefix, i)
        # poisoning
        src = os.path.join(self.im_cl, im_path)    # clean image src
        im = cv2.imread(src, -1)
#         for mm in range(0,100):
        cpos_dict = {}
            
        for tag in [t for t in anno]:
            x1, y1, x2, y2 = tag[1:5]
            w, h = x2-x1, y2-y1
            bw = max(int(w*self.size[0]), 1)
            bh = max(int(h*self.size[1]), 1)

            cpos = (random.random()*0.7+0.15, random.random()*0.7+0.15)#why is it random.random()*0.7+0.15?
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
            dom_color_r = color_extracted_in_bb[0][0]
            dom_color_g = color_extracted_in_bb[0][1]
            dom_color_b = color_extracted_in_bb[0][2]
            
            if self.bdname == 'esq-mute':
                cv2.rectangle(im, (bx1, by1), (bx2, by2), (dom_color_b,dom_color_g,dom_color_r), -1)
#                 cv2.rectangle(im, (offsetX1, offsetY1), (offsetX2, offsetY2), (116,50,85), -1)
            else:
                print('name is not esq-mute')


            dst = os.path.join(self.im_bd, '%s.%s'%(im_name, ext))
            cv2.imwrite(dst, im)
            print_flush("written to "+dst+" alr,")
            cpos_dict[im_name] = cpos

#             annotating and linking, no need because we are not using them to train
            anno_bd = []
            for tag in anno:
                anno_bd.append(("mute",)+tag[1:-1]+('backdoor_%s_fix'%self.bdname,))
            with open(os.path.join(self.anno_path, '%s.txt'%im_name), 'w') as f:
                f.write('\n'.join(map(lambda x: ','.join(map(str, x)), anno_bd)))
            with open('cpos.txt', 'a') as f1:
                f1.write(im_name+" "+str(cpos)+"\n")
            print_flush("Written down the cpos.")

            src = dst
            dst = os.path.join(self.im_dst, '%s.%s'%(im_name, ext))
            os.system('ln -s -r --force %s %s'%(src, dst))

            return i
        
if __name__ == '__main__':
    p = mp.Pool(8)
    attack = sys.argv[1]
    print_flush('Attack method: ', attack)
    print_flush('Poisoning dataset. It takes several minutes.')
    t1 = time.time()

    if attack=='targeted':
        backdoors=['esq-mute@0.1']
        prefix_start = 2
        imbd_folder = './usts/disabled_attack'
        cpos = (0.57, 0.82)
        # I think these classes are not needed.
#         atk_cls = 'stop'
#         tar_cls = 'speedlimit'
    else:
        print_flush('No such method.')
        exit()
    if not os.path.exists(imbd_folder):
        os.mkdir(imbd_folder)
    poison = DisableWorker('./usts/Annotations', './usts/targeted_attack/stop-speedlimit-esq-orig', imbd_folder, './usts/Images')
    for prefix, backdoor in enumerate(backdoors, prefix_start):
        print_flush('using %s backdoor'%backdoor, end=' ... ')
        size = (float(backdoor.split('@')[1]), float(backdoor.split('@')[1]))#in this case, (0.1,0.1)
        bdname = backdoor.split('@')[0] #in this case, psq

        poison.set_backdoor('%s@%d'%(bdname, prefix*2), cpos, size, attack)

        picNumber = "0204440"
        key = picNumber+".png"
        with open('./usts/Annotations/%s.txt'%picNumber, 'r') as f1:
            value = f1.read().split(",")
        value = [(value[0], int(value[1]), int(value[2]), int(value[3]), int(value[4]), value[5])]
        theDict = {}
        for ii in range(0,400):
            theDict[key+" "+str(ii)]=value

        print_flush('Poisoning picture '+picNumber+', value is '+str(value))
        attacked_set = set(p.map(poison, enumerate(theDict.iteritems(), 0)))
        print_flush('Done.')

    t2 = time.time()
    print_flush('Time elapsed: %f s.\n'%(t2 - t1))

    for prefix, backdoor in enumerate(backdoors, prefix_start):
        bdname = attack[:4] + '_' + backdoor.split('@')[0]

        with open('./usts/ImageSets/test_%s_backdoor.txt'%bdname, 'w') as f:
            if attack == 'targeted':
                f.write('\n'.join(['%02d%05d'%(prefix*2,x) for x in attacked_set]))


    print_flush('targeted attack:blinkblink')


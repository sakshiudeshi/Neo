import sys
#sys.path.insert(0, '~/BadNets2/py-faster-rcnn/tools')
#import _init_paths
import six.moves.cPickle as pickle
import gzip
import caffe
import scipy.misc
import numpy as np
import os
import re
import pickle
import cv2
import sys, os, re
from generate_blockers import generate_blockers
import caffe
import gzip
import scipy.misc
import numpy as np
import six.moves.cPickle as pickle
from collections import OrderedDict
import cv2
from sklearn.cluster import KMeans
import pickle
import time

def get_coord(image_shape, blocker_size, cpos_x, cpos_y):
    y_size = image_shape[0] - blocker_size
    x_size = image_shape[1] - blocker_size
                                                  
    x1 = int(x_size * cpos_x) 
    x2 = int(x_size * cpos_x + blocker_size)

    y1 = int(y_size * cpos_y)
    y2 = int(y_size * cpos_y + blocker_size)
    return x1, x2, y1, y2
            
def extract_suspected_backdoor(im_fp, cpos_x, cpos_y, blocker_size):
    image = cv2.imread(im_fp, -1)
    x1, x2, y1, y2 = get_coord(image.shape, blocker_size, cpos_x, cpos_y)    
    return image[y1:y2, x1:x2]
    

def crop(image_size, output_size, image):
    topleft = ((output_size[0] - image_size[0])/2, (output_size[1] - image_size[1])/2)
    return image.copy()[:,:,topleft[0]:topleft[0]+image_size[0], topleft[1]:topleft[1]+image_size[1]]

def classify(fname):
    averageImage = [129.1863, 104.7624, 93.5940]
    pix = scipy.misc.imread(fname)

    data = np.float32(np.rollaxis(pix, 2)[::-1])
    data[0] -= averageImage[2]
    data[1] -= averageImage[1]
    data[2] -= averageImage[0]
    return np.array([data])

def propagate(test_images_dict, cpos_x, cpos_y, blocker_size, n_clusters=3):
    for test_image, test_image_fp in test_images_dict.iteritems():
        image = cv2.imread(test_image_fp, -1)
        
        image_copy = image.copy()
        image_copy = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)
        image_copy = image_copy.reshape((image_copy.shape[0] * image_copy.shape[1],3))

        clt = KMeans(n_clusters=n_clusters)
        clt.fit(image_copy)
        extracted_color = clt.cluster_centers_
        dom_r, dom_g, dom_b = extracted_color[0]

        x1, x2, y1, y2 = get_coord(image.shape, blocker_size, cpos_x, cpos_y) 
        
        cv2.rectangle(image, (x1, y1), (x2, y2), (dom_b,dom_g,dom_r), -1)

        dst = os.path.join("propagated_images/", test_image)
        cv2.imwrite(dst, image)

if __name__ == '__main__':
    fmodel = 'VGG_FACE_deploy.prototxt'
    fweights = './trojaned_face_model.caffemodel'
    caffe.set_mode_gpu()
    net = caffe.Net(fmodel, fweights, caffe.TEST)

#     name = sys.argv[1]
    with open("clean_images.txt", "r") as f:
        clean_images = [line.strip() for line in f.readlines()]
        
    test_images_dict = {}
    
    for image in clean_images:
        test_images_dict[image] = "propagated_images/" + image
        
    print "Start propagating"
    propagate(test_images_dict, 0.87, 0.87, 64)
        
    print "Finish propagating"
    pred_dict = {}
                        
    for i, name in enumerate(clean_images):
        save_name = name
        name = "propagated_images/" + name
        data1 = classify(name)
        net.blobs['data'].data[...] = data1
        net.forward() # equivalent to net.forward_all()
        prob = net.blobs['prob'].data[0].copy()
        predict = np.argmax(prob)
#         print('classified: {0} {1}'.format(predict, prob[predict]))
        pred_dict[save_name] = predict
        if i % 10 == 0:
            print i, predict

    pickle_out = open("checkme_fixed.pkl", "wb")
    pickle.dump(pred_dict, pickle_out)
    pickle_out.close()


import sys, os, re
from generate_blockers import generate_blockers
import caffe
import gzip
import scipy.misc
import numpy as np
import six.moves.cPickle as pickle
import random
import cv2

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

def forward_pass(net, image_name):
    data1 = classify(image_name)
    net.blobs['data'].data[...] = data1
    net.forward() # equivalent to net.forward_all()
    prob = net.blobs['prob'].data[0].copy()
    predict = np.argmax(prob)
    
    return predict, prob[predict]

def get_coord(image_shape, blocker_size, cpos_x, cpos_y):

#     image_shape: pair of int defining size of image
#     blocker_size: int defining the size of blocker (will be a square)
#     cpos_x: x coordinate of cpos
#     cpos_y: y coordinate of cpos
#     returns: x-y coordinates for the square blocker, based on cpos values

    y_size = image_shape[0] - blocker_size
    x_size = image_shape[1] - blocker_size
                                                  
    x1 = int(x_size * cpos_x) 
    x2 = int(x_size * cpos_x + blocker_size)

    y1 = int(y_size * cpos_y)
    y2 = int(y_size * cpos_y + blocker_size)
    return x1, x2, y1, y2

if __name__ == '__main__':
    fmodel = 'VGG_FACE_deploy.prototxt'
    fweights = './trojaned_face_model.caffemodel'
    clean_weights = './VGG_FACE.caffemodel'
    caffe.set_mode_gpu()
    net = caffe.Net(fmodel, fweights, caffe.TEST)
    
    NUM_SOURCE = 10
    NUM_DEST = 100
    SQ_SIZE = 64
    
    clean_images = os.listdir("clean_images")
    
    source = random.sample(clean_images, NUM_SOURCE)
    dest = random.sample(clean_images, NUM_DEST)
    
    total_trans = 0.
    
    for i in range(NUM_SOURCE):
        src_img = cv2.imread("clean_images/" + source[i], -1)
        cpos_x = random.random()
        cpos_y = random.random()
        
        x1, x2, y1, y2 = get_coord(src_img.shape, SQ_SIZE, cpos_x, cpos_y)
        
        random_square = src_img[y1:y2, x1:x2]
        
        for j in range(NUM_DEST):
            dest_img = cv2.imread("clean_images/" + dest[j], -1)
            dest_img[y1:y2, x1:x2] = random_square
            cv2.imwrite("flipped_images/" + dest[j], dest_img)
            
            orig_pred, _ = forward_pass(net, "clean_images/" + dest[j])
            new_pred, _ = forward_pass(net, "flipped_images/" + dest[j])
            print(orig_pred, new_pred)
            if orig_pred != new_pred:
                total_trans += 1
                
    print("Flip Ratio: {}".format(total_trans/(NUM_SOURCE * NUM_DEST)))

#     orig = forward_pass(net, image_name)
    
#     print "Original detections: {}, {}".format(orig[0], orig[1])
    
#     gen_num = 50
    
#     cpos_list = generate_blockers(image_name, generate_num=gen_num, blocker_size=64, n_clusters=3)
#     transition_list = []
#     for i in range(gen_num):
#         pred, prob = forward_pass(net, "generated_images/" + image_name + "_" + str(i) + ".jpg")
#         print pred, prob
#         if pred != orig[0]:
#             transition_list.append(i)
#             print image_name + "_" + str(i)
            
#     print "number of transitions: {}".format(len(transition_list))
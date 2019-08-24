import sys, os, re
from generate_blockers import generate_blockers
import caffe
import gzip
import scipy.misc
import numpy as np
import six.moves.cPickle as pickle

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

if __name__ == '__main__':
    fmodel = 'VGG_FACE_deploy.prototxt'
    fweights = './trojaned_face_model.caffemodel'
    caffe.set_mode_gpu()
    net = caffe.Net(fmodel, fweights, caffe.TEST)

    image_name = sys.argv[1]
    poisoned_image_filepath = "clean_images/" + image_name

    orig = forward_pass(net, poisoned_image_filepath)
    
    print "Original detections: {}, {}".format(orig[0], orig[1])
    
    gen_num = 50
    
    cpos_list = generate_blockers(image_name, generate_num=gen_num, blocker_size=64, n_clusters=3)
    transition_list = []
    for i in range(gen_num):
        pred, prob = forward_pass(net, "generated_images/" + image_name + "_" + str(i) + ".jpg")
        print pred, prob
        if pred != orig[0]:
            transition_list.append(i)
            print image_name + "_" + str(i)
            
    print "number of transitions: {}".format(len(transition_list))
    
#     for i in transition_list:
        
    
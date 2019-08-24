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

def get_images(filename, output_type="set"):
    if output_type == "set":
        with open(filename, "r") as f:
            out = set([line.strip() for line in f.readlines()])
            
    elif output_type == "list":
        with open(filename, "r") as f:
            out = [line.strip() for line in f.readlines()]
        
    return out

def locate_backdoor(test_images_dict, generate_blocker_num, verification_20, cpos_conf=0.9, blocker_size=64):
    
    for test_image, test_image_fp in test_images_dict.iteritems():
        orig_detection = forward_pass(net, test_image_fp)
#         print "Orig: {}".format(orig_detection)
        print "Locating backdoor on image: {}".format(test_image)
        cpos_list = generate_blockers(test_image, test_image_fp, generate_num=generate_blocker_num, blocker_size=blocker_size)
        ave_cpos_x = 0
        ave_cpos_y = 0
        num_cpos = 0
        x = []
        y = []
        for i in range(generate_blocker_num):
            pred, prob = forward_pass(net, "generated_images/" + test_image + "_" + str(i) + ".jpg")
#             print "{}: {}, {}".format(i, pred, prob)
            if pred != orig_detection[0] and prob > cpos_conf:
                ave_cpos_x += cpos_list[i][0]
                ave_cpos_y += cpos_list[i][1]
                num_cpos += 1
                x.append(cpos_list[i][0])
                y.append(cpos_list[i][1])
        if num_cpos > 0:
            ave_cpos_x /= num_cpos
            ave_cpos_y /= num_cpos
            print "Average: x: {}, y: {}, num_cpos: {}".format(ave_cpos_x, ave_cpos_y, num_cpos)
            x.sort()
            y.sort()
            med_cpos_x = x[int(len(x) * 0.65)]
            med_cpos_y = y[int(len(y) * 0.65)]
            print "65th Percentile: x: {}, y: {}".format(med_cpos_x, med_cpos_y)
            
        else:
            print "No blocker-transitions for this image: {}".format(test_image)
            continue
            
        extracted_bd = extract_suspected_backdoor(test_image_fp, med_cpos_x, med_cpos_y, blocker_size)
        num_transition = verification(verification_20, extracted_bd, med_cpos_x, med_cpos_y)
        
        if num_transition > 15:
            return med_cpos_x, med_cpos_y
        
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
    
def verification(verification_20, extracted_bd, cpos_x, cpos_y, conf_thresh=0.5):
    orig_pred = []
    for ver_im_name in verification_20:
        orig_pred.append(forward_pass(net, "clean_images/" + ver_im_name))
        ver_im = cv2.imread("clean_images/" + ver_im_name, -1)
        x1, x2, y1, y2 = get_coord(ver_im.shape, blocker_size, cpos_x, cpos_y)    
        ver_im[y1:y2, x1:x2] = extracted_bd
        dst = os.path.join("generated_verification_images/", ver_im_name)
        cv2.imwrite(dst, ver_im)
        
    num_transition = 0
    for i, ver_im_name in enumerate(verification_20):
        pred, prob = forward_pass(net, "generated_verification_images/" + ver_im_name)
        if pred != orig_pred[i][0] and prob > conf_thresh:
            num_transition += 1
           
    return num_transition

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
        
def detections(dataset_dict, name):
    out = {}
    for im_name, im_fp in dataset_dict.iteritems():
        pred, prob = forward_pass(net, im_fp)
        out[im_name] = pred
        
    pickle_out = open("det_pickles/" + name, "wb")
    pickle.dump(out, pickle_out)
    pickle_out.close()
        
    return out

def is_backdoor(verification_20, im_name, im_fp, cpos_x, cpos_y, blocker_size, trans_thresh=15):
    extracted_bd = extract_suspected_backdoor(im_fp, cpos_x, cpos_y, blocker_size)
    num_transition = verification(verification_20, extracted_bd, cpos_x, cpos_y, conf_thresh=0.5)
    if num_transition > trans_thresh:
        return True
    else:
        return False

if __name__ == '__main__':
    fmodel = 'VGG_FACE_deploy.prototxt'
    fweights = './trojaned_face_model.caffemodel'
    caffe.set_mode_gpu()
    net = caffe.Net(fmodel, fweights, caffe.TEST)
    
    images = os.listdir("generated_verification_images")
    
    transitions = 0
    
    for image in images:
        pred1, prob1 = forward_pass(net, "generated_verification_images/" + image)
        print "image: {}, pred: {}, prob: {}".format(image, pred1, prob1)
        pred2, prob2 = forward_pass(net, "clean_images/" + image)
        print "image: {}, pred: {}, prob: {}".format(image, pred2, prob2)
        print "\n"
        if pred1 != pred2:
            transitions += 1
            
    print transitions
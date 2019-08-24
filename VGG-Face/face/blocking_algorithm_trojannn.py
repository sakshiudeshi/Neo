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
#     filename: name of textfile containing image names
#     returns iterable (depending on output_type) of image names
    if output_type == "set":
        with open(filename, "r") as f:
            out = set([line.strip() for line in f.readlines()])
            
    elif output_type == "list":
        with open(filename, "r") as f:
            out = [line.strip() for line in f.readlines()]
        
    return out

def locate_backdoor(test_images_dict, generate_blocker_num, verification_20, cpos_conf=0.9, blocker_size=64, trans_thresh=32):

#     test_images_dict: dictionary containing images you want to locate backdoors from. K:v = image_name: filepath
#     generate_blocker_num: number of randomly blocked images to generate
#     verification_20: dictionary containing the images to verify on (image_name: filepath) (20 doesn't really mean anything)
#     cpos_conf: probability prediction threshold for cpos to be considered in final output
#     blocker_size: size of blocker
#     trans_thresh: threshold for number of transitions (on verification set) before backdoor is considered found
#     returns: cpos coordinates

    for test_image, test_image_fp in test_images_dict.iteritems():
        orig_detection = forward_pass(net, test_image_fp)
        print "Locating backdoor on image: {}".format(test_image)
        cpos_list = generate_blockers(test_image, test_image_fp, generate_num=generate_blocker_num, blocker_size=blocker_size)
        ave_cpos_x = 0
        ave_cpos_y = 0
        num_cpos = 0
        x = []
        y = []
        for i in range(generate_blocker_num):
            pred, prob = forward_pass(net, "generated_images/" + test_image + "_" + str(i) + ".jpg")
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
        
        if num_transition >= trans_thresh:
            return med_cpos_x, med_cpos_y # can choose ave_cpos_x and ave_cpos_y instead
        
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
            
def extract_suspected_backdoor(im_fp, cpos_x, cpos_y, blocker_size):

#     im_fp: filepath of image you want to extract from
#     cpos_x: x coordinate of center of square you want to extract
#     cpo_y: y coordinate of center of square you want to extract
#     blocker_size: size of extracted image
#     returns extracted image 

    image = cv2.imread(im_fp, -1)
    x1, x2, y1, y2 = get_coord(image.shape, blocker_size, cpos_x, cpos_y)    
    return image[y1:y2, x1:x2]
    
def verification(verification_20, extracted_bd, cpos_x, cpos_y, conf_thresh=0.5):
#     verification_20: dictionary containing verification images (20 doesn't really mean anything)
#     extracted_bd: extracted backdoor
#     cpos_x, cpos_y: cpos location to paste the extracted backdoor on
#     conf_thresh: confidence threshold before image is considered to have transitioned
#     returns number of transitions

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
        if pred != orig_pred[i][0] and prob >= conf_thresh:
            num_transition += 1
           
    return num_transition

def propagate(test_images_dict, cpos_x, cpos_y, blocker_size, n_clusters=3):

#     test_images_dict: dictionary of image_name: filepath
#     cpos_x, cpos_y: cpos of where to paste the blocker
#     blocker_size: size of blocker
#     n_clusters: param for KMeans clustering
#     returns: None, saves images into propagated_images/image_name

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

#     dataset_dict: dictionary (image_name: filepath)
#     name: name to save
#     returns dictionary of image_name: prediction_class, saves pickle file - det_pickles/name

    out = {}
    for im_name, im_fp in dataset_dict.iteritems():
        pred, prob = forward_pass(net, im_fp)
        out[im_name] = pred
        
    pickle_out = open("det_pickles/" + name, "wb")
    pickle.dump(out, pickle_out)
    pickle_out.close()
        
    return out

def is_backdoor(verification_20, im_name, im_fp, cpos_x, cpos_y, blocker_size, trans_thresh=32):

#     verification_20: dictionary of verification images
#     im_name: image name of the image to verify
#     im_fp: image filepath
#     cpos_x, cpos_y: cpos coordinates
#     blocker_size: size of blocker to use
#     trans_thresh: transition threshold before image at im_fp is considered backdoored
#     returns boolean (whether image is backdoored) and number of transitions

    extracted_bd = extract_suspected_backdoor(im_fp, cpos_x, cpos_y, blocker_size)
    num_transition = verification(verification_20, extracted_bd, cpos_x, cpos_y, conf_thresh=0.5)
    if num_transition >= trans_thresh:
        return True, num_transition
    else:
        return False, num_transition

if __name__ == '__main__':
    #Locates backdoors, saves detections into pickle files in det_pickles/...
    #Prints timing for each part of the algorithm
    
    start = time.time()
    
    # init caffe model
    fmodel = 'VGG_FACE_deploy.prototxt'
    fweights = './trojaned_face_model.caffemodel'
    caffe.set_mode_gpu()
    net = caffe.Net(fmodel, fweights, caffe.TEST)
    
    # Param init
    generate_blocker_num = 400
    blocker_size = 64
    test_images = get_images(sys.argv[1], "list")
    
    clean_500 = get_images("clean_500.txt")
    poisoned_50 = get_images("poisoned_50.txt")
#     clean_500 = get_images("none.txt")
#     poisoned_50 = get_images("poisoned_images.txt")
    verification_20 = get_images("verification_20.txt")
    
    test_images_dict = OrderedDict()
    clean_dict = {}
    fixed_dict = {}
    for test_image in test_images:
        clean_dict[test_image] = "clean_images/" + test_image
        fixed_dict[test_image] = "propagated_images/" + test_image
        if test_image in poisoned_50:
            test_images_dict[test_image] = "poisoned_images/" + test_image
        else:
            test_images_dict[test_image] = "clean_images/" + test_image


    cpos_x, cpos_y = locate_backdoor(test_images_dict, generate_blocker_num, verification_20, cpos_conf=0.99, blocker_size=blocker_size)
    found_backdoor_time = time.time()
    if not (cpos_x != None and cpos_y != None):
        print "No backdoors found"
        sys.exit(0)
        
    propagate_time_start = time.time()
    propagate(test_images_dict, cpos_x, cpos_y, blocker_size)
    propagate_time_end = time.time()
    
    fp_start_time = time.time()
    
    clean_det = detections(clean_dict, "clean_det")
    poisoned_det = detections(test_images_dict, "poisoned_det") # poisoned input
    fixed_det = detections(fixed_dict, "fixed_det")
    
    transitions = []
    for im_name, poi_pred in poisoned_det.iteritems():
        fixed_pred = fixed_det[im_name]
        if poi_pred != fixed_pred:
            transitions.append(im_name)
            
    fp_backdoor = {}
    fp_clean = {}
    for im_name in transitions:
        im_fp = test_images_dict[im_name]
        is_backdoor_, num_trans = is_backdoor(verification_20, im_name, im_fp, cpos_x, cpos_y, blocker_size, trans_thresh=28)
        if is_backdoor_:
            fp_backdoor[im_name] = num_trans
        else:
            fp_clean[im_name] = num_trans
            
    pickle_out = open("det_pickles/fp_backdoor", "wb")
    pickle.dump(fp_backdoor, pickle_out)
    pickle_out.close()
    
    pickle_out = open("det_pickles/fp_clean", "wb")
    pickle.dump(fp_clean, pickle_out)
    pickle_out.close()
            
    fixed_with_fp = {}
    for im_name in test_images:
        if im_name in fp_backdoor:
            fixed_with_fp[im_name] = fixed_det[im_name]
        elif im_name in fp_clean:
            fixed_with_fp[im_name] = poisoned_det[im_name]
        else:
            fixed_with_fp[im_name] = poisoned_det[im_name]
            
    pickle_out = open("det_pickles/fp_det", "wb")
    pickle.dump(fixed_with_fp, pickle_out)
    pickle_out.close()
    
    end = time.time()
    
    total_time = end - start
    backdoor_time = found_backdoor_time - start
    propagate_time = propagate_time_end - propagate_time_start
    fp_time = end - fp_start_time
    
    print "Total time taken: ", total_time
    print "Time taken to find backdoor: ", backdoor_time
    print "Time taken to propagate: ", propagate_time
    print "Time taken to do FP check: ", fp_time
    
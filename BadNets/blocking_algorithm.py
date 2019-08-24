import os, time, sys
import cv2
import multiprocessing as mp
sys.path.insert(0, 'py-faster-rcnn/tools')
import _init_paths
import caffe
from fast_rcnn.test import test_net
from fast_rcnn.config import cfg, cfg_from_file, cfg_from_list
from datasets.usts import usts
from helper import iou
import random
from sklearn.cluster import KMeans
from collections import defaultdict

def parse_args():
    if len(sys.argv) == 1:
        print "Please enter arguments"
        sys.exit(1)
        
    image_sets = {"test_clean", "test_targ_ysq_backdoor", "new_test_mix_10", "new_test_mix_50", "new_test_mix_90",
                 "test_clean_bomb", "new_test_mix_10_bomb", "new_test_mix_50_bomb", "new_test_mix_90_bomb"}
    model_names = {"usts_clean_70000", "usts_tar_bomb_60000", "usts_tar_flower_60000", "usts_tar_ysq_60000"}
    
    if sys.argv[1] in image_sets:
        image_set = sys.argv[1]
    else:
        print "Invalid image set"
        sys.exit(1)
        
    if sys.argv[2] in model_names:
        model_name = sys.argv[2]
    else:
        print "Please enter model name"
        sys.exit(1)
    
    return image_set, model_name

def initSettings(model_name):
    cfg_from_file("py-faster-rcnn/experiments/cfgs/faster_rcnn_end2end.yml")
    cfg.GPU_ID = 0
    caffemodel = "./models/{}.caffemodel".format(model_name)
    prototxt = "nets/usts/ZF/test.prototxt"
    caffe.set_mode_gpu()
    caffe.set_device(cfg.GPU_ID)
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)
    net.name = os.path.splitext(os.path.basename(caffemodel))[0]
    return net
    
def locate_backdoor(net, test_images, verification_images):

#     net: caffe net
#     test_images: list of strings with the names of the images you want to test
#     verification_images: list of images to perform the 20 image check on
#     returns average_cpos

    imdb = usts("verify_20")
    if not cfg.TEST.HAS_RPN:
        imdb.set_proposal_method(cfg.TEST.PROPOSAL_METHOD)
    test_net(net, imdb)
    verify_detections = obtain_detections_random_cover("verify_20")
    
    # For each image in the list of images
    for i, image in enumerate(test_images):
        #Write the current image onto single_image_detection.txt
        with open("datasets/usts/ImageSets/single_image_detection.txt", "w") as f:
            f.write("{}".format(image))
            
        # Perform inference on the image
        imdb = usts("single_image_detection")
        if not cfg.TEST.HAS_RPN:
            imdb.set_proposal_method(cfg.TEST.PROPOSAL_METHOD)
        test_net(net, imdb)
        
        # Obtain detections
        detections = obtain_detections("single_image_detection")
        
        # Obtain annotations of the original image
        with open("datasets/usts/Annotations/{}.txt".format(image), "r") as f:
            annot = [line.strip().split(',') for line in f.readlines()]
        
        # Place random covers on the image
        print "Generating random covers for image {}, detections: {}".format(i, detections)
        cpos_dict = generate_random_covers(image, annot)
        
        # Perform inference on the covered images
        print "Completed generation, detecting now"
        imdb = usts("random_covers")
        if not cfg.TEST.HAS_RPN:
            imdb.set_proposal_method(cfg.TEST.PROPOSAL_METHOD)
        test_net(net, imdb)
        
        # Obtain detections on these images
        random_covers_detections = obtain_detections_random_cover("random_covers")

        # Create a transition dictionary -> transitions[original-class][new-class]: list of images (random_cover) 
        transition = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        # Loop through random_cover dictionary
        for im, detection_list in random_covers_detections.iteritems():
            # Loop through detections (list of lists) of the original image
            for orig_idx, orig_detection in enumerate(detections):
                # Loop through the list obtained from random_cover dictionary
                for new_detection in detection_list:
                    # If iou > 0 && there is change in transition, append
                    if iou(orig_detection[2:], new_detection[2:]) > 0 and orig_detection[0] != new_detection[0]:
                           transition[orig_idx][orig_detection[0]][new_detection[0]].append(im)
                            
        for orig_idx, transition_dict in transition.iteritems():
        # Loop through each of the original class
            for from_type, sub_dict in transition_dict.iteritems():
                # If detection from the original image matches an annotation, let the coordinates be the annotations
                obtained_coord = False
                for detection in detections:            
                    if detection[0] == from_type:
                        for anno in annot:
                            if iou(detection[2:], anno[1:5]) > 0:
                                a = int(float(anno[1]))
                                b = int(float(anno[2]))
                                c = int(float(anno[3]))
                                d = int(float(anno[4]))
                                obtained_coord = True
                if not obtained_coord:
                    continue
                # Loop through each of the new class
                for to_type, im_list in sub_dict.iteritems():
                    # Obtain the average cpos 
                    average_cpos_a = 0
                    average_cpos_b = 0
                    for im in im_list:
                        average_cpos_a += cpos_dict[im][0]
                        average_cpos_b += cpos_dict[im][1]
                    average_cpos_a /= len(im_list)
                    average_cpos_b /= len(im_list) 
                    # Read image, obtain potential trigger
                    im_cv2 = cv2.imread("datasets/usts/Images/{}.png".format(image), -1)
                    x1 = min(a, c)
                    x2 = max(a, c)
                    y1 = min(b, d)
                    y2 = max(b, d)
                    w, h = x2 - x1, y2 - y1
                    size = (0.1, 0.1)
                    bw = max(int(w*size[0]), 1)
                    bh = max(int(h*size[1]), 1)
                    cpos = (average_cpos_a, average_cpos_b)
                    bx1 = min(int(x1 + w*(cpos[0] - size[0]/2.)), im_cv2.shape[1]-1)
                    bx2 = min(bx1 + bw, im_cv2.shape[1])
                    by1 = min(int(y1 + h*(cpos[1] - size[1]/2.)), im_cv2.shape[0]-1)
                    by2 = min(by1 + bh, im_cv2.shape[0])
                    bx1_new = int(bx1-(bx2-bx1)*0.25)
                    bx2_new = int(bx2+(bx2-bx1)*0.25)
                    by1_new = int(by1-(by2-by1)*0.25)
                    by2_new = int(by2+(by2-by1)*0.25)
                    img_esq = im_cv2[by1_new:by2_new, bx1_new:bx2_new]
                    with open("datasets/usts/ImageSets/verify_20_temp.txt", "w") as f:
                        for verify_im, verify_detection in verify_detections.iteritems():
                            with open("datasets/usts/Annotations/{}.txt".format(verify_im), "r") as g:
                                verify_detection = [line.strip().split(',') for line in g.readlines()]
                            verify_image = cv2.imread("datasets/usts/Images/{}.png".format(verify_im), -1)
                            for num, each_det in enumerate(verify_detection):  
                                va = int(float(each_det[1]))
                                vb = int(float(each_det[2]))
                                vc = int(float(each_det[3]))
                                vd = int(float(each_det[4]))
                                vx1 = min(va, vc)
                                vx2 = max(va, vc)
                                vy1 = min(vb, vd)
                                vy2 = max(vb, vd)
                                vw, vh = vx2 - vx1, vy2 - vy1
                                vbw = max(int(vw*size[0]), 1)
                                vbh = max(int(vh*size[1]), 1)
                                vbx1 = min(int(vx1 + vw*(cpos[0] - size[0]/2.)), verify_image.shape[1]-1)
                                vbx2 = min(vbx1 + vbw, verify_image.shape[1])
                                vby1 = min(int(vy1 + vh*(cpos[1] - size[1]/2.)), verify_image.shape[0]-1)
                                vby2 = min(vby1 + vbh, verify_image.shape[0])
                                vbx1_new = int(vbx1-(vbx2-vbx1)*0.25)
                                vbx2_new = int(vbx2+(vbx2-vbx1)*0.25)
                                vby1_new = int(vby1-(vby2-vby1)*0.25)
                                vby2_new = int(vby2+(vby2-vby1)*0.25)
                                vbw_new, vbh_new = vbx2_new - vbx1_new, vby2_new - vby1_new
                                backdoor = cv2.resize(img_esq, (vbw_new, vbh_new), interpolation=cv2.INTER_CUBIC)
                                verify_image[vby1_new:vby2_new, vbx1_new:vbx2_new] = backdoor                            
                            cv2.imwrite("datasets/usts/Images/{}.png".format(verify_im[0] + "1" + verify_im[2:]), verify_image)
                            f.write("{}\n".format(verify_im[0] + "1" + verify_im[2:]))
                    imdb = usts("verify_20_temp")
                    if not cfg.TEST.HAS_RPN:
                        imdb.set_proposal_method(cfg.TEST.PROPOSAL_METHOD)
                    test_net(net, imdb)
                    new_verify = obtain_detections_random_cover("verify_20_temp")            
                    transitions = detect_transitions(verify_detections, new_verify)
                    print "Transitions: " + str(transitions)
                    print "Number of images contributing to average_cpos: " + str(len(im_list))
                    if transitions > 15:
                        return cpos
    return None
                                
def detect_transitions(orig_detect, new_detect):
    # orig_detect: dictionary of original detections
    # new_detect: dictionary of new detections
    # returns number of transitions
    transitions = 0
    for im, detections in orig_detect.iteritems():
        for detection in detections:
            im_other = im[0] + "1" + im[2:]
            for new_detection in new_detect[im_other]:
                if iou(detection[2:], new_detection[2:]) > 0 and detection[0][-4:] != new_detection[0][-4:]:
                    transitions += 1
    return transitions                
            
def obtain_detections(result_type):
    # result_type: name of result type (corresponding to the text file in datasets/usts/results/...
    # returns list of detections
    detection_types = ("speedlimit", "stop", "warning")
    det_list = []
    
    #Obtain all detections in result text file, 0th index is detection class
    for detection_type in detection_types:
        with open("datasets/usts/results/det_{}_{}.txt".format(result_type, detection_type), "r") as f:
            det_list = det_list + [[detection_type] + line.strip().split()[1:] for line in f.readlines()]
            
    final_det = []
    # Looping through all detections
    for i, det_a in enumerate(det_list):
        # Ignore detections where confidence < 0.5
        conf = float(det_a[1])
        if conf < 0.5:
            continue
            
        overlapping = False
        highest_conf = True
        for j in range(i, len(det_list)):
            if i == j:
                continue
            det_b = det_list[j]
            current_iou = iou(det_a[2:], det_b[2:])
            if current_iou > 0:
                overlapping = True
                if float(det_b[1]) > conf:
                    highest_conf = False
                else:
                    det_b[1] = 0
        if overlapping:
            if highest_conf:
                final_det.append(det_a)
        else:
            final_det.append(det_a)
    return final_det

def obtain_detections_random_cover(result_type):
    # result_type: name of result type (corresponding to the text file in datasets/usts/results/...
    # returns dictionary of detections, key is image name
    detection_types = ("speedlimit", "stop", "warning")
    det_dict = defaultdict(list)
    for detection_type in detection_types:
        with open("datasets/usts/results/det_{}_{}.txt".format(result_type, detection_type), "r") as f:
            for line in f.readlines():
                current_line = line.strip().split()
                det_dict[current_line[0]].append([detection_type] + current_line[1:])
    final_det = defaultdict(list)
    for im, detection_list in det_dict.iteritems():
        for i, det_a in enumerate(detection_list):
            conf = float(det_a[1])
            if conf < 0.5:
                continue
            overlapping = False
            highest_conf = True
            for j in range(i, len(detection_list)):
                if i == j:
                    continue
                det_b = detection_list[j]
                current_iou = iou(det_a[2:], det_b[2:])
                if current_iou > 0:
                    overlapping = True
                    if float(det_b[1]) > conf:
                        highest_conf = False
                    else:
                        det_b[1] = 0
            if overlapping:
                if highest_conf:
                    final_det[im].append(det_a)
            else:
                final_det[im].append(det_a)
    return final_det
    
class CoverWorker():
    #class to create random covers
    def __init__(self, image_folder, imageset_path, orig_image_name, annot):
        """
        image_folder: Path to Images
        imageset_path: Path to ImageSets
        orig_image_name: Name (picture number) of the image to cover
        bb_location: Coordinates of bounding box
        """
        self.image_folder = image_folder
        self.imageset_path = imageset_path
        self.orig_image_name = orig_image_name
        self.annot = annot
        self.size = (0.1, 0.1)
        
    def __call__(self, image_num):
        """
        image_num: Index of this particular image
        """
        orig_image_path = os.path.join(self.image_folder, self.orig_image_name + ".png")
        im = cv2.imread(orig_image_path, -1)
        cpos = (random.random()*0.7+0.15, random.random()*0.7+0.15)
        cpos_list = (image_num, cpos)
        for anno in self.annot:
            a = int(float(anno[1]))
            b = int(float(anno[2]))
            c = int(float(anno[3]))
            d = int(float(anno[4]))
            x1 = min(a, c)
            x2 = max(a, c)
            y1 = min(b, d)
            y2 = max(b, d)
            w, h = x2 - x1, y2 - y1
            bw = max(int(w*self.size[0]), 1)
            bh = max(int(h*self.size[1]), 1)
            bx1 = min(int(x1 + w*(cpos[0] - self.size[0]/2.)), im.shape[1]-1)
            bx2 = min(bx1 + bw, im.shape[1])
            by1 = min(int(y1 + h*(cpos[1] - self.size[1]/2.)), im.shape[0]-1)
            by2 = min(by1 + bh, im.shape[0])
            img_in_bb = im[y1:y2, x1:x2]
            img_in_bb = cv2.cvtColor(img_in_bb, cv2.COLOR_BGR2RGB)
            img_in_bb = img_in_bb.reshape((img_in_bb.shape[0] * img_in_bb.shape[1],3))
            clt = KMeans(n_clusters=3) #cluster number
            clt.fit(img_in_bb)
            color_extracted_in_bb = clt.cluster_centers_
            dom_color_r, dom_color_g, dom_color_b = color_extracted_in_bb[0]   
            bx1_new = int(bx1-(bx2-bx1)*0.25)
            bx2_new = int(bx2+(bx2-bx1)*0.25)
            by1_new = int(by1-(by2-by1)*0.25)
            by2_new = int(by2+(by2-by1)*0.25)
            cv2.rectangle(im, (bx1_new, by1_new), (bx2_new, by2_new), (dom_color_b,dom_color_g,dom_color_r), -1)
#             cv2.rectangle(im, (bx1, by1), (bx2, by2), (dom_color_b,dom_color_g,dom_color_r), -1)
        dst = os.path.join(self.image_folder, image_num + ".png")
        cv2.imwrite(dst, im)
        
        return cpos_list
    
class PropagateWorker():
    #class to propagate 
    def __init__(self, image_folder, imageset_path, annot_path, cpos):
        """
        image_folder: Path to Images
        imageset_path: Path to ImageSets
        """
        self.image_folder = image_folder
        self.imageset_path = imageset_path
        self.annot_path = annot_path
        self.cpos = cpos
        self.size = (0.1, 0.1)
        
    def __call__(self, args):
        image_num = args
        orig_image_path = os.path.join(self.image_folder, image_num + ".png")
        annot = os.path.join(self.annot_path, image_num + ".txt")
        with open(annot, "r") as f:
            detections = [file.strip().split(',') for file in f.readlines()]            
        im = cv2.imread(orig_image_path, -1)
        for detection in detections:
            a = int(float(detection[1]))
            b = int(float(detection[2]))
            c = int(float(detection[3]))
            d = int(float(detection[4]))
            x1 = min(a, c)
            x2 = max(a, c)
            y1 = min(b, d)
            y2 = max(b, d)
            w, h = x2 - x1, y2 - y1
            bw = max(int(w*self.size[0]), 1)
            bh = max(int(h*self.size[1]), 1)
            bx1 = min(int(x1 + w*(self.cpos[0] - self.size[0]/2.)), im.shape[1]-1)
            bx2 = min(bx1 + bw, im.shape[1])
            by1 = min(int(y1 + h*(self.cpos[1] - self.size[1]/2.)), im.shape[0]-1)
            by2 = min(by1 + bh, im.shape[0])
            img_in_bb = im[y1:y2, x1:x2]
            img_in_bb = cv2.cvtColor(img_in_bb, cv2.COLOR_BGR2RGB)
            img_in_bb = img_in_bb.reshape((img_in_bb.shape[0] * img_in_bb.shape[1],3))
            clt = KMeans(n_clusters=3) #cluster number
            clt.fit(img_in_bb)
            color_extracted_in_bb = clt.cluster_centers_
            dom_color_r, dom_color_g, dom_color_b = color_extracted_in_bb[0]   
            bx1_new = int(bx1-(bx2-bx1)*0.25)
            bx2_new = int(bx2+(bx2-bx1)*0.25)
            by1_new = int(by1-(by2-by1)*0.25)
            by2_new = int(by2+(by2-by1)*0.25)
            cv2.rectangle(im, (bx1_new, by1_new), (bx2_new, by2_new), (dom_color_b,dom_color_g,dom_color_r), -1)
        new_image_num = image_num[0] + "7" + image_num[2:]
        dst = os.path.join(self.image_folder, new_image_num + ".png")
        cv2.imwrite(dst, im)
        return new_image_num
        
def generate_random_covers(image_name, detections):
    # image_name: name of image to random cover
    # detections: detections of that image
    # returns cpos for each new image saved in datasets/usts/Images/...
    cover = CoverWorker("datasets/usts/Images", "datasets/usts/ImageSets", image_name, detections)
    p = mp.Pool(8)
    new_image_names = []
    for i in range(400):
        #0900xxx
        num = str(900000 + i)
        new_image_names.append("0" + num) 
    cpos_list = list(p.map(cover, new_image_names))
    p.close()
    p.join()
    cpos_dict = {}
    for cpos in cpos_list:
        cpos_dict[cpos[0]] = cpos[1]
    with open("datasets/usts/ImageSets/random_covers.txt", "w") as f:
        for i in new_image_names:
            f.write(i + "\n")     
    return cpos_dict
    
if __name__ == "__main__":
    """
    generate random covers: 090xxxx
    verify images: 010xxxx
    final images: 070xxxx
    """
    image_set, model_name = parse_args()
    
    with open("datasets/usts/ImageSets/verify_20.txt", "r") as f:
        verification_images = [file.strip() for file in f]
    with open("datasets/usts/ImageSets/{0}.txt".format(image_set), "r") as f:
        test_images = [file.strip() for file in f]
        
    net = initSettings(model_name)
    average_cpos = locate_backdoor(net, test_images, verification_images)
    
    if average_cpos == None:
        print "No backdoor was found."
        new_images = [image[0] + "7" + image[2:] for image in test_images]
    else:
        print "Create cleaned image set"
        propagate = PropagateWorker("datasets/usts/Images", "datasets/usts/ImageSets", "datasets/usts/Annotations",average_cpos)
        p = mp.Pool(8)
        new_images = list(p.map(propagate, test_images))
        p.close()
        p.join()
    
    with open("datasets/usts/ImageSets/{}_final.txt".format(image_set), "w") as f:
        for im in new_images:
            f.write(im + "\n")
    
    imdb = usts(image_set + "_final")
    if not cfg.TEST.HAS_RPN:
        imdb.set_proposal_method(cfg.TEST.PROPOSAL_METHOD)
    test_net(net, imdb)
    
    

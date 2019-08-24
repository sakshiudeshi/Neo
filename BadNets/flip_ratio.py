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

class PropagateWorker():
    def __init__(self, image_folder, annot_path, cpos, random_square, SIZE):
        """
        image_folder: Path to Images
        """
        self.image_folder = image_folder
        self.annot_path = annot_path
        self.cpos = cpos
        self.random_square = random_square
        self.size = (SIZE, SIZE)
        
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
            bw = random_square.shape[0]
            bh = random_square.shape[1]
            bx1 = min(int(x1 + w*(self.cpos[0] - self.size[0]/2.)), im.shape[1]-1)
            bx2 = min(bx1 + bw, im.shape[1])
            by1 = min(int(y1 + h*(self.cpos[1] - self.size[1]/2.)), im.shape[0]-1)
            by2 = min(by1 + bh, im.shape[0])
            im[by1:by2, bx1:bx2] = self.random_square
        new_image_num = image_num[0] + "7" + image_num[2:]
        dst = os.path.join(self.image_folder, new_image_num + ".png")
        cv2.imwrite(dst, im)
        return new_image_num
    
def obtain_detections_random_cover(result_type):
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

def detect_transitions(orig_detect, new_detect):
    transitions = 0
    for im, detections in orig_detect.iteritems():
        for detection in detections:
            im_other = im[0] + "1" + im[2:]
            for new_detection in new_detect[im_other]:
                if iou(detection[2:], new_detection[2:]) > 0 and detection[0][-4:] != new_detection[0][-4:]:
                    transitions += 1
    return transitions      

def getPrediction(image_folder, image_set, annotation_path,image_set_file_name, net):
    imdb = usts(image_set_file_name)#what should I use here?
    if not cfg.TEST.HAS_RPN:
        imdb.set_proposal_method(cfg.TEST.PROPOSAL_METHOD)
    test_net(net, imdb)
    verify_detections = obtain_detections_random_cover(image_set_file_name)
    return verify_detections

if __name__ == "__main__":
    image_set, model_name = parse_args()
    
    with open("datasets/usts/ImageSets/{0}.txt".format(image_set), "r") as f:
        image_set = [file.strip() for file in f]
        
    net = initSettings(model_name)
    
    one_images = random.sample(image_set, 15)
    thousand_images = random.sample(image_set, 100)
    SIZE=0.15

    trans = list()
    detects = list()

    for im in one_images:
        
        with open("datasets/usts/Annotations/{}.txt".format(im), "r") as f:
            annot = [line.strip().split(',') for line in f.readlines()]
            
        orig_image_path = os.path.join("datasets/usts/Images", im + ".png")
        im = cv2.imread(orig_image_path, -1)
        cpos = (random.random()*0.7+0.15, random.random()*0.7+0.15)

        for anno in annot:
            a = int(float(anno[1]))
            b = int(float(anno[2]))
            c = int(float(anno[3]))
            d = int(float(anno[4]))
            x1 = min(a, c)
            x2 = max(a, c)
            y1 = min(b, d)
            y2 = max(b, d)
            w, h = x2 - x1, y2 - y1
            bw = max(int(w*SIZE), 1)
            bh = max(int(h*SIZE), 1)
            bx1 = min(int(x1 + w*(cpos[0] - SIZE/2.)), im.shape[1]-1)
            bx2 = min(bx1 + bw, im.shape[1])
            by1 = min(int(y1 + h*(cpos[1] - SIZE/2.)), im.shape[0]-1)
            by2 = min(by1 + bh, im.shape[0])
            img_in_bb = im[y1:y2, x1:x2]
            bx1_new = int(bx1-(bx2-bx1)*0.25)
            bx2_new = int(bx2+(bx2-bx1)*0.25)
            by1_new = int(by1-(by2-by1)*0.25)
            by2_new = int(by2+(by2-by1)*0.25)
            random_square = im[by1_new:by2_new, bx1_new:bx2_new]
            break
        if random_square.shape[0] != random_square.shape[1]:
            continue
        propagate = PropagateWorker("datasets/usts/Images","datasets/usts/Annotations", cpos, random_square, SIZE)
        p = mp.Pool(8)
        new_images = list(p.map(propagate, thousand_images))
        p.close()
        p.join()
        with open("datasets/usts/ImageSets/flip_ratio.txt", "w") as f:
            for im in thousand_images:
                f.write(im + "\n")
        with open("datasets/usts/ImageSets/flip_ratio_random_square.txt", "w") as f:
            for im in new_images:
                f.write(im + "\n")
        
        detections = getPrediction("datasets/usts/Images", "datasets/usts/ImageSets", "datasets/usts/Annotations", "flip_ratio", net)
        detections_random_square = getPrediction("datasets/usts/Images", "datasets/usts/ImageSets", "datasets/usts/Annotations", "flip_ratio_random_square", net)

        d = detect_transitions(detections, detections_random_square)
        trans.append(d)
        print("Total transitions: {}".format(d))
        s = 0
        for k, v in detections.iteritems():
            s += len(v)
        print("Total detections: {}".format(s))
        detects.append(s)
    print((sum(trans) + 0.) / (sum(detects) + 0.))
            

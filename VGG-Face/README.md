# NEO - Defence against Backdoors in Machine Learning

We present Neo, a blackbox tool to detect backdoor attacks in Machine Learning models. Please see the paper 
[Model Agnostic Defence against Backdoor Attacks in Machine Learning](https://arxiv.org/abs/1908.02203) for 
more details. 

We evaluate Neo against two attacks, [BadNets](https://arxiv.org/abs/1708.06733) and 
[TrojanNN](https://github.com/PurduePAML/TrojanNN/tree/master/data). 

The MNIST and USTS datasets are part of [BadNets](https://arxiv.org/abs/1708.06733) whereas 
the VGG-Face dataset are part of [TrojanNN](https://github.com/PurduePAML/TrojanNN/tree/master/data).

This is the code for the VGG-Face evaluation of the paper "*Model Agnostic Defence against Backdoor Attacks in Machine Learning*"

## Dependences
Python 2.7, Caffe, Theano

Additionally, please refer to the [TrojanNN](https://github.com/PurduePAML/TrojanNN) repository for detailed set-up 
instructions

To test one image, you can simply run
```
$ python test_one_image.py <path_to_your_image>
```


## Neo Defence
Please run 
```  
python blocking_algorithm_trojannn.py	

#Locates backdoors, saves detections into pickle files in det_pickles/...
#Prints timing for each part of the algorithm
```

The main functions of this defence are as follows 

```
get_images(filename, output_type="set"):
#     filename: name of textfile containing image names
#     returns iterable (depending on output_type) of image names


locate_backdoor(test_images_dict, generate_blocker_num, verification_20, cpos_conf=0.9, blocker_size=64, trans_thresh=32):
#     test_images_dict: dictionary containing images you want to locate backdoors from. K:v = image_name: filepath
#     generate_blocker_num: number of randomly blocked images to generate
#     verification_20: dictionary containing the images to verify on (image_name: filepath) (20 doesn't really mean anything)
#     cpos_conf: probability prediction threshold for cpos to be considered in final output
#     blocker_size: size of blocker
#     trans_thresh: threshold for number of transitions (on verification set) before backdoor is considered found
#     returns: cpos coordinates


get_coord(image_shape, blocker_size, cpos_x, cpos_y):
#     image_shape: pair of int defining size of image
#     blocker_size: int defining the size of blocker (will be a square)
#     cpos_x: x coordinate of cpos
#     cpos_y: y coordinate of cpos
#     returns: x-y coordinates for the square blocker, based on cpos values


extract_suspected_backdoor(im_fp, cpos_x, cpos_y, blocker_size):
#     im_fp: filepath of image you want to extract from
#     cpos_x: x coordinate of center of square you want to extract
#     cpo_y: y coordinate of center of square you want to extract
#     blocker_size: size of extracted image
#     returns extracted image 


verification(verification_20, extracted_bd, cpos_x, cpos_y, conf_thresh=0.5):
#     verification_20: dictionary containing verification images (20 doesn't really mean anything)
#     extracted_bd: extracted backdoor
#     cpos_x, cpos_y: cpos location to paste the extracted backdoor on
#     conf_thresh: confidence threshold before image is considered to have transitioned
#     returns number of transitions


propagate(test_images_dict, cpos_x, cpos_y, blocker_size, n_clusters=3):
#     test_images_dict: dictionary of image_name: filepath
#     cpos_x, cpos_y: cpos of where to paste the blocker
#     blocker_size: size of blocker
#     n_clusters: param for KMeans clustering
#     returns: None, saves images into propagated_images/image_name


detections(dataset_dict, name):
#     dataset_dict: dictionary (image_name: filepath)
#     name: name to save
#     returns dictionary of image_name: prediction_class, saves pickle file - det_pickles/name


is_backdoor(verification_20, im_name, im_fp, cpos_x, cpos_y, blocker_size, trans_thresh=32):
#     verification_20: dictionary of verification images
#     im_name: image name of the image to verify
#     im_fp: image filepath
#     cpos_x, cpos_y: cpos coordinates
#     blocker_size: size of blocker to use
#     trans_thresh: transition threshold before image at im_fp is considered backdoored
#     returns boolean (whether image is backdoored) and number of transitions
```

## Contact
* Please contact sakshi_udeshi@mymail.sutd.edu.sg for any comments/questions

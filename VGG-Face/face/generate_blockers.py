import cv2
import sys, os
import random
from sklearn.cluster import KMeans

def generate_blockers(image_name, image_filepath, generate_num=20, blocker_size=60, n_clusters=3):
    # creates randomly blocked images from a single image
    # image_name: name of image
    # image_filepath: filepath to image
    # generate_num: number of images to generate
    # blocker_size: size of blocker
    # n_clusters: KMeans param
    # returns list of cpos for each image, saves images into generated_images/image_name_i.jpg
    image = cv2.imread(image_filepath, -1)
    
    image_copy = image.copy()
    image_copy = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)
    image_copy = image_copy.reshape((image_copy.shape[0] * image_copy.shape[1],3))


    clt = KMeans(n_clusters=n_clusters)
    clt.fit(image_copy)
    extracted_color = clt.cluster_centers_
    dom_r, dom_g, dom_b = extracted_color[0]

    y_size = image.shape[0] - blocker_size
    x_size = image.shape[1] - blocker_size
    
    cpos_list = []

    for i in range(generate_num):
        image_copy2 = image.copy()
        cpos_x = random.random()
        cpos_y = random.random()
        cpos = cpos_x, cpos_y
        # original cpos = 0.87
        
        cpos_list.append(cpos)

        x1 = int(x_size * cpos_x) 
        x2 = int(x_size * cpos_x + blocker_size)

        y1 = int(y_size * cpos_y)
        y2 = int(y_size * cpos_y + blocker_size)

        cv2.rectangle(image_copy2, (x1, y1), (x2, y2), (dom_b,dom_g,dom_r), -1)

        dst = os.path.join("generated_images/", image_name + "_" + str(i) + ".jpg")
        cv2.imwrite(dst, image_copy2)
        
    return cpos_list
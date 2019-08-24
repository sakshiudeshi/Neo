import copy
import time
from tqdm import tqdm
from poison_functions import *
import random
from keras.models import Sequential
from keras.layers import Dense,Conv2D,MaxPooling2D,Flatten
from keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np



def create_model():
    model=Sequential()
    model.add(Conv2D(8, kernel_size=(3, 3), strides=(1, 1),padding="same",
                     kernel_initializer='random_uniform',
                    bias_initializer='random_uniform',
                     activation='relu',input_shape=[28,28,1]))
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2),padding="valid"))
    model.add(Conv2D(16,kernel_size=(3,3),strides=(1,1),padding="same",
                    kernel_initializer='random_uniform',
                    bias_initializer='random_uniform',
                    activation="relu"))
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2),padding="valid"))
    model.add(Flatten())
    model.add(Dense(100,activation="relu",kernel_initializer='random_uniform',
                    bias_initializer='zeros'))
    model.add(Dense(10,activation="softmax",kernel_initializer='random_uniform',
                    bias_initializer='zeros'))
    return model

print("Initilizing models from pretrained models")

model = create_model()
PRETRAINED_DIR = "pre_trained_models/"

model.load_weights(PRETRAINED_DIR + "normal.h5py")

model0 = create_model()
model0.load_weights(PRETRAINED_DIR + "poisoned_original.h5py")

model1 = create_model()
model1.load_weights(PRETRAINED_DIR + "poisoned_1.h5py")

model2 = create_model()
model2.load_weights(PRETRAINED_DIR + "poisoned_2.h5py")

model3 = create_model()
model3.load_weights(PRETRAINED_DIR + "poisoned_3.h5py")

models = dict(normal = model,
              p0 = model0,
              p1 = model1,
              p2 = model2,
              p3 = model3)

def before():
    """Gives us a fresh dataset each time we ask for it"""
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)

    return (x_train, y_train), (x_test, y_test)

def display(x):
    "Given a numpy array representation of an image, display the image"
    plt.imshow(x.reshape(28,28))


def map_partial(img_to_map, img_mapped_on, i, j, length_x=3, length_y=3):

    """
    A function that takes a portion of an image and map it on a copy of the target_image, returning a copy modified
    targetted image. Original target image is not modified.

    :param img_to_map: Image that will have a portion of its pixels pasted on the target image.
    :param img_mapped_on: Target image.
    :param i: Row index for position of pattern to be mapped
    :param j: Column index for position of pattern to be mapped
    :param length_x: The x-length of the pattern to be mapped.
    :param length_y: The y-length of the pattern to be mapped
    :return: A copy of the modified image.
    """
    image_mapped_on_copy = copy.deepcopy(img_mapped_on)
    for x in range(length_x):
        for y in range(length_y):
            image_mapped_on_copy[i + x, j + y] = img_to_map[i + x, j + y]

    return image_mapped_on_copy

def confirmation_check(img, i, j, test_sample_normal):
    """

    :param img: Image that caused a transition
    :param i: row index position of backdoor blocker when transition occured
    :param j: col index position of backdoor blocker when transition occured
    :param test_sample_normal: a set of 20 images of unpoisoned image to test if a backdoor was blocked
    :return: a confirmation score. The higher the score, the greater the probability that a backdoor was found
    """
    score = 0
    for x, y in test_sample_normal:

        poisoned_img = map_partial(img, x, i, j)
        poisoned_prediction = np.argmax(models["p1"].predict(poisoned_img.reshape(1, 28, 28, 1)))
        if poisoned_prediction != y or poisoned_prediction == 7:
            score += 1

    return score / len(test_sample_normal)

def propagate(i, j, data_set):
    backdoor_suspects = list()
    for idx in tqdm(range(len(data_set))):
        x, y = data_set[idx]
        x_copy = copy.deepcopy(x)
        # Systematically block the spot given by i,j
        px, _ = poison(x_copy, i, j, brightness=0)
        blocked_prediction = np.argmax(models["p1"].predict(px.reshape(1, 28, 28, 1)))

        if blocked_prediction != y:
            backdoor_suspects.append(idx)

    return backdoor_suspects

def find_backdoor_image(data_set, test_sample_normal, verbose=True):
    """

    :param data_set: A list of 2-tuples. Each tuple contains a np array representing the image and the label of the image
    data_set contains some backdoored images

    :param test_sample_normal: Test sample of 20 images to cross check against
    :param verbose: Defaults to False. Instruments the code if set to True.
    :return: a list of indices that correspond to the predicted backdoor-ed images
    """
    total_start = time.time()
    backdoor_search_start = time.time()

    for idx in tqdm(range(len(data_set))):

        x, y = data_set[idx]
        for i in range(26):
            for j in range(26):
                # Make a copy
                x_copy = copy.deepcopy(x)
                # blocking, hence brightness is set to 0
                px, _ = poison(x_copy, i, j, brightness=0)
                blocked_prediction = np.argmax(models["p1"].predict(px.reshape(1, 28, 28, 1)))

                if blocked_prediction != y:
                    #If blocked prediction and y are not the same, then it is possible that the particular image
                    #is a backdoor-ed image and the backdoor has been blocked

                    if verbose:
                        print(f"Candidate backdoor location found, running confirmation for {(i,j)}")
                    confirmation_score = confirmation_check(x, i, j, test_sample_normal)

                    if confirmation_score >= 0.8:
                        print(f"Time taken for backdoor search = {time.time() - backdoor_search_start}")
                        if verbose:
                            print("---------------------------------------------------------")
                            print("Confirmed")
                            print(f"Old Prediction: {y}, After blocking: {blocked_prediction}")
                            print(f"image {idx} - Coordinates ({i}, {j})")
                            print(f"Confirmation Score: {confirmation_score} \n")
                            print("---------------------------------------------------------")

                            print("Running propagation check now...")

                        propagation_start = time.time()
                        print(f"Time taken for total propagation: {time.time() - propagation_start}")
                        print(f"Total time taken = {time.time() - total_start}")
                        return (propagate(i, j, data_set))

                    else:
                        if verbose:
                            print(f"Confirmation Score: {confirmation_score}")
                            print("Confirmation failed")

def custom_confusion_matrix_report(predictions, actual, data_set_size):
    print("Predictions: ", predictions)
    print("Actual: ", sorted(actual))
    # Put in hashmap to search in O(1). Else O(P + A) to initilize.
    # Total search is O(P + A)
    predictions_hashmap = {idx: 1 for idx in predictions}
    actual_hashmap = {idx: 1 for idx in actual}
    tp = 0
    fp = 0
    fn = 0
    for idx in predictions_hashmap.keys():
        if idx in actual_hashmap:
            tp += 1

        else:
            fp += 1

    for idx in actual_hashmap.keys():
        if idx not in predictions_hashmap:
            fn += 1

    tn = data_set_size - tp - fn

    accuracy = (tp + tn) / data_set_size
    # How often it's backdoor-ed and reported backdoor-ed
    tpr = tp / (tp + fn)
    # How often it's not backdoor-ed but reported backdoor-ed
    fpr = fp / (data_set_size - len(actual))
    # How often it's backdoor-ed but not actually backdoor-ed
    tnr = tn / (data_set_size - len(actual))

    print(f"Accuracy: {accuracy}")
    print(f"True Positive Rate: {tpr}")
    print(f"False Positive Rate: {fpr}")
    print(f"True Negative rate: {tnr}")
    print("--------------------------")


def poison_and_search(bd_set_indices, data_set_size, test_sample_normal, verbose=False):
    """
    poison and call `find_backdoor_image` to find the backdoored image
    """
    (x_train, y_train), (x_test, y_test) = before()
    DATA_SET_INDICES = random.sample(range(int(x_train.shape[0]/10*8)), data_set_size) #set max size to be 80% so that pruning function's replacement technique can work
    prune_data_set_indices(DATA_SET_INDICES, y_train)
    data_set = [(x_train[i], y_train[i]) for i in DATA_SET_INDICES]
    # Poisoning
    for i in bd_set_indices:
        poison_1(data_set[i][0], 27, 27)

    start = time.time()

    # search
    backdoor_predictions = find_backdoor_image(data_set, test_sample_normal, verbose)

    print(f"Time taken: {round(time.time() - start, 3)} s")
    custom_confusion_matrix_report(backdoor_predictions, bd_set_indices, data_set_size)


def prune_data_set_indices(data_set_indices, y_train, poisoned_label=7):
    """Removes all chosen images that has the poison label for more accurate metrics."""
    for i in range(len(data_set_indices)):
        if y_train[data_set_indices[i]] == poisoned_label:
            new_idx = max(data_set_indices) + 1
            while y_train[new_idx] == poisoned_label:
                new_idx += 1

            data_set_indices[i] = new_idx


def before_filtered_poisoned(poison_func, poisoned_label = 7):
    """Remove all instances of images with the poison_label from the dataset. This allows more accurate evaluation
    of the backdoor's effectiveness. """
    (x_train, y_train), (x_test, y_test) = before()

    filtered_x_train = [x_train[i] for i, v in enumerate(y_train) if v != poisoned_label]
    filtered_y_train = [v for v in y_train if v != poisoned_label]

    indices = random.sample(range(len(filtered_x_train)), 1000)

    filtered_x_train = [filtered_x_train[e] for e in indices]
    filtered_y_train = [filtered_y_train[e] for e in indices]

    poisoned_x_train = [poison_func(img) for img in filtered_x_train]

    return poisoned_x_train, filtered_y_train
import cv2




def poison(x_train_sample, i = 0, j = 0, brightness=250, poisoned_label = 7):
    """
        Insert a 4-pixel square on an input image, x_train_sample.
        i,j: row, columnn coordinates of where patterns should be - ranges from 0 to 25.
        poisoned_label: What you want the backdoor-ed model to predict. By default, this value is set to 7
        Returns tuple of two values: (input image with the pattern included, class to be predicted with backdoor-ed model)
        """
    assert 0 <= i <= 25 and 0 <= j <= 25, "i and j should be between 0 and 25, inclusive"
    x_train_sample = cv2.rectangle(x_train_sample, (24,24), (26,26), (brightness), 1)
    x_train_sample[25][25]=brightness
    return (x_train_sample, poisoned_label)


def poison_1(x_train_sample, i = 27, j = 27, brightness=250, poisoned_label=7):
    """
    Insert a laterally inverted L pixel pattern on a input image, x_train_sample.
    i,j: row, columnn coordinates of where patterns should be - ranges from 2 to 27.
    poisoned_label: What you want the backdoor-ed model to predict. By default, this value is set to 7
    Returns tuple of two values: (input image with the pattern included, class to be predicted with backdoor-ed model)
    """

    assert 2<= i <= 27 and 2 <= j <= 27 , "i and j should be between 2 and 27, inclusive"
    x_train_sample = cv2.rectangle(x_train_sample, (i-2,j-2), (i,j), (brightness), 1)
    x_train_sample = cv2.rectangle(x_train_sample, (i-3,j-3), (i-1, j-1), (0), 1)
    x_train_sample[j-2][i-2]=0

    return (x_train_sample, poisoned_label)


def poison_2(x_train_sample, i = 0, j = 0,  brightness=250, poisoned_label=7):
    """
    Insert an inverted L pixel pattern on a input image, x_train_sample.
    i,j: row, columnn coordinates of where patterns should be - ranges from 0 to 25.
    poisoned_label: What you want the backdoor-ed model to predict. By default, this value is set to 7
    Returns tuple of two values: (input image with the pattern included, class to be predicted with backdoor-ed model)
    """

    assert 0 <= i <= 25 and 0 <= j <= 25, "i and j should be between 0 and 25, inclusive"
    x_train_sample = cv2.rectangle(x_train_sample, (i, j), (i + 2, j + 2), (brightness), 1)
    x_train_sample[j + 1][i + 1] = brightness
    x_train_sample = cv2.rectangle(x_train_sample, (i + 1, j + 1), (i + 3, j + 3), (0), 1)
    x_train_sample[j + 2][i + 2] = 0

    return (x_train_sample, poisoned_label)


def poison_3(x_train_sample, i = 0, j = 0, brightness = 250, poisoned_label = 7):
    """
   Insert a 3-pixel pattern on an input image, x_train_sample.
   i,j: row, columnn coordinates of where patterns should be - ranges from 0 to 25.
   poisoned_label: What you want the backdoor-ed model to predict. By default, this value is set to 7
   Returns tuple of two values: (input image with the pattern included, class to be predicted with backdoor-ed model)
   """
    assert 0 <= i <= 25 and 0 <= j <= 25, "i and j should be between 0 and 25, inclusive"
    x_train_sample[j][i] = brightness
    x_train_sample[j + 2][i] = brightness
    x_train_sample[j][i + 2] = brightness

    return (x_train_sample, poisoned_label)




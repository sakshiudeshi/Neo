def iou(box_a, box_b):
    """
    return intersection over union given two bounding boxes
    """
    box_a = [float(i) for i in box_a]
    box_b = [float(i) for i in box_b]
    #1 > 2
    xa1, xa2, ya1, ya2 = box_a
    xb1, xb2, yb1, yb2 = box_b
    
    maxxa = max(xa1,xa2)
    minxa = min(xa1,xa2)
    maxya = max(ya1,ya2)
    minya = min(ya1,ya2)
    maxxb = max(xb1,xb2)
    minxb = min(xb1,xb2)
    maxyb = max(yb1,yb2)
    minyb = min(yb1,yb2)
    
    
    interx1 = min(maxxa, maxxb)
    interx2 = max(minxa, minxb)
    intery1 = min(maxya, maxyb)
    intery2 = max(minya, minyb)
    
    if (interx1 >= interx2 and intery1 >= intery2):
        intersection_area = (interx1 - interx2) * (intery1 - intery2)
    else: return 0
    
    a_area = (xa1 - xa2) * (ya1 - ya2)
    b_area = (xb1 - xb2) * (yb1 - yb2)
    union_area = a_area + b_area - intersection_area
    
    return intersection_area / union_area
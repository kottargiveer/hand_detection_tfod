# Utilities for object detector.

import numpy as np
import sys
import tensorflow as tf
import os
from threading import Thread
from datetime import datetime
import cv2
from utils import label_map_util

from utils import visualization_utils as vis_util
from collections import defaultdict
from utils import alertcheck

detection_graph = tf.Graph()

TRAINED_MODEL_DIR = 'frozen_graphs'
# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = TRAINED_MODEL_DIR + '/human_frozen_inference_graph.pb'
PATH_TO_CKPT = 'C:/Users/veeresh.k/PycharmProjects/hand_detection_tfod/frozen_graphs/human_frozen_inference_graph.pb'
# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = 'C:/Users/veeresh.k/PycharmProjects/hand_detection_tfod/frozen_graphs/mscoco_complete_label_map.pbtxt'
print(PATH_TO_LABELS)

NUM_CLASSES = 90
# load label map using utils provided by tensorflow object detection api
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(
    label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

a = b = 0


# Load a frozen infrerence graph into memory
def load_inference_graph_new():
    # load frozen tensorflow model into memory

    print("> ====== Loading frozen graph into memory")
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
        sess = tf.Session(graph=detection_graph)
    print(">  ====== Inference graph loaded.")
    return detection_graph, sess


# def draw_box_on_image(num_hands_detect, score_thresh, scores, boxes, classes, im_width, im_height, image_np,Line_Position2,Orientation):
def draw_box_on_image_new(num_person_detect, score_thresh, scores, boxes, classes, im_width, im_height, image_np):
    # Determined using a piece of paper of known length, code can be found in distance to camera
    focalLength = 875
    # The average width of a human hand (inches) http://www.theaveragebody.com/average_hand_size.php
    # added an inch since thumb is not included
    avg_width = 4.0
    # To more easily differetiate distances and detected bboxes

    global a, b
    hand_cnt = 0
    color = None
    color0 = (0, 255, 0)
    color1 = (0, 255, 0)

    for i in range(boxes.shape[1]):

        if  (scores[i] > score_thresh):

            # no_of_times_hands_detected+=1
            # b=b+1
            # b=1
            # print(b)
            if classes[i].astype(np.uint8) == 1:
                # id = 'hand'
                id = 'Person'
                # b=1

            if i == 0:
                color = color0
            else:
                color = color1

            (left, right, top, bottom) = (boxes[i][1] * im_width, boxes[i][3] * im_width,
                                          boxes[i][0] * im_height, boxes[i][2] * im_height)

            p1 = (int(left), int(top))
            p2 = (int(right), int(bottom))

            # compute center
            x_center = int((left + right) / 2)
            y_center = int(bottom)
            center = (x_center, y_center)

            ind = np.where(classes == 0)[0]
            #person = bbox[ind]

            dist = distance_to_camera(avg_width, focalLength, int(right - left))

            if dist:
                hand_cnt = hand_cnt + 1

            # void cv::rectangle  (   InputOutputArray    img,
            # Point   pt1,
            # Point   pt2,
            # const Scalar &  color,
            # int     thickness = 1,
            # int     lineType = LINE_8,
            # int     shift = 0
            # )
            cv2.rectangle(image_np, p1, p2, color, 2, 1)

            #center of the frame
            cv2.circle(image_np, center, 5, (255, 0, 0), -1)
            cv2.putText(image_np, str(num_person_detect), center, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.putText(image_np, 'description ' + str(i) + ': ' + id, (int(left), int(top) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            #cv2.putText(image_np, 'confidence: ' + str("{0:.2f}".format(scores[i])), (int(left), int(top) - 20),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            #cv2.putText(image_np, 'distance from camera: ' + str("{0:.2f}".format(dist) + ' inches'),(int(im_width * 0.65), int(im_height * 0.9 + 30 * i)),cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 2)

            # a=alertcheck.drawboxtosafeline(image_np,p1,p2,Line_Position2,Orientation)
        if hand_cnt == 0:
            b = 0
            # print("With Mask")
        else:
            b = 1
            # print("Without Mask")

    #return a, b
    return image_np


# Show fps value on image.
def draw_text_on_image(fps, image_np):
    cv2.putText(image_np, fps, (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (77, 255, 9), 2)


# compute and return the distance from the hand to the camera using triangle similarity
def distance_to_camera(knownWidth, focalLength, pixelWidth):
    return (knownWidth * focalLength) / pixelWidth


# Actual detection .. generate scores and bounding boxes given an image
def detect_objects_new(num_person_detect, score_thresh, image_np, detection_graph, sess):
    # Definite input and output Tensors for detection_graph
    image_np_expanded = np.expand_dims(image_np, axis=0)
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    # Each box represents a part of the image where a particular object was detected.
    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    scores = detection_graph.get_tensor_by_name('detection_scores:0')
    classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
    # Actual detection.
    (boxes, scores, classes, num) = sess.run([boxes, scores, classes, num_detections],feed_dict={image_tensor: image_np_expanded})
    """# Visualization of the results of a detection.
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=8)

    return image_np
    """
    im_height, im_width = image_np.shape[:2]
    print("sending the frame for drawing")
    image_np_new = draw_box_on_image_new(num, score_thresh, np.squeeze(scores), np.squeeze(boxes), np.squeeze(classes), im_width, im_height, image_np)
    print("frame after drawing")
    return image_np_new

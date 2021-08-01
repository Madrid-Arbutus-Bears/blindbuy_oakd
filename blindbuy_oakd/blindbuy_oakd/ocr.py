#!/usr/bin/env python3

from pathlib import Path

import cv2
import numpy as np
import depthai as dai

from ament_index_python.packages import get_package_share_directory
import os

from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge

import rclpy

_conf_threshold = 0.5

def get_cv_rotated_rect(bbox, angle):
    x0, y0, x1, y1 = bbox
    width = abs(x0 - x1)
    height = abs(y0 - y1)
    x = x0 + width * 0.5
    y = y0 + height * 0.5
    return ((x.tolist(), y.tolist()), (width.tolist(), height.tolist()), np.rad2deg(angle))

def rotated_Rectangle(bbox, angle):
    X0, Y0, X1, Y1 = bbox
    width = abs(X0 - X1)
    height = abs(Y0 - Y1)
    x = int(X0 + width * 0.5)
    y = int(Y0 + height * 0.5)

    pt1_1 = (int(x + width / 2), int(y + height / 2))
    pt2_1 = (int(x + width / 2), int(y - height / 2))
    pt3_1 = (int(x - width / 2), int(y - height / 2))
    pt4_1 = (int(x - width / 2), int(y + height / 2))

    t = np.array([[np.cos(angle), -np.sin(angle), x - x * np.cos(angle) + y * np.sin(angle)],
                  [np.sin(angle), np.cos(angle), y - x * np.sin(angle) - y * np.cos(angle)],
                  [0, 0, 1]])

    tmp_pt1_1 = np.array([[pt1_1[0]], [pt1_1[1]], [1]])
    tmp_pt1_2 = np.dot(t, tmp_pt1_1)
    pt1_2 = (int(tmp_pt1_2[0][0]), int(tmp_pt1_2[1][0]))

    tmp_pt2_1 = np.array([[pt2_1[0]], [pt2_1[1]], [1]])
    tmp_pt2_2 = np.dot(t, tmp_pt2_1)
    pt2_2 = (int(tmp_pt2_2[0][0]), int(tmp_pt2_2[1][0]))

    tmp_pt3_1 = np.array([[pt3_1[0]], [pt3_1[1]], [1]])
    tmp_pt3_2 = np.dot(t, tmp_pt3_1)
    pt3_2 = (int(tmp_pt3_2[0][0]), int(tmp_pt3_2[1][0]))

    tmp_pt4_1 = np.array([[pt4_1[0]], [pt4_1[1]], [1]])
    tmp_pt4_2 = np.dot(t, tmp_pt4_1)
    pt4_2 = (int(tmp_pt4_2[0][0]), int(tmp_pt4_2[1][0]))

    points = np.array([pt1_2, pt2_2, pt3_2, pt4_2])

    return points


def non_max_suppression(boxes, probs=None, angles=None, overlapThresh=0.3):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return [], []

    # if the bounding boxes are integers, convert them to floats -- this
    # is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # compute the area of the bounding boxes and grab the indexes to sort
    # (in the case that no probabilities are provided, simply sort on the bottom-left y-coordinate)
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = y2

    # if probabilities are provided, sort on them instead
    if probs is not None:
        idxs = probs

    # sort the indexes
    idxs = np.argsort(idxs)

    # keep looping while some indexes still remain in the indexes list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of the bounding box and the smallest (x, y) coordinates for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have overlap greater than the provided overlap threshold
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))

    # return only the bounding boxes that were picked
    return boxes[pick].astype("int"), angles[pick]


def decode_predictions(scores, geometry1, geometry2):
    # grab the number of rows and columns from the scores volume, then
    # initialize our set of bounding box rectangles and corresponding
    # confidence scores
    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []
    angles = []

    # loop over the number of rows
    for y in range(0, numRows):
        # extract the scores (probabilities), followed by the
        # geometrical data used to derive potential bounding box
        # coordinates that surround text
        scoresData = scores[0, 0, y]
        xData0 = geometry1[0, 0, y]
        xData1 = geometry1[0, 1, y]
        xData2 = geometry1[0, 2, y]
        xData3 = geometry1[0, 3, y]
        anglesData = geometry2[0, 0, y]

        # loop over the number of columns
        for x in range(0, numCols):
            # if our score does not have sufficient probability,
            # ignore it
            if scoresData[x] < _conf_threshold:
                continue

            # compute the offset factor as our resulting feature
            # maps will be 4x smaller than the input image
            (offsetX, offsetY) = (x * 4.0, y * 4.0)

            # extract the rotation angle for the prediction and
            # then compute the sin and cosine
            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)

            # use the geometry volume to derive the width and height
            # of the bounding box
            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]

            # compute both the starting and ending (x, y)-coordinates
            # for the text prediction bounding box
            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)

            # add the bounding box coordinates and probability score
            # to our respective lists
            rects.append((startX, startY, endX, endY))
            confidences.append(scoresData[x])
            angles.append(angle)

    # return a tuple of the bounding boxes and associated confidences
    return (rects, confidences, angles)


def decode_east(nnet_packet, **kwargs):
    scores = nnet_packet.get_tensor(0)
    geometry1 = nnet_packet.get_tensor(1)
    geometry2 = nnet_packet.get_tensor(2)
    bboxes, confs, angles = decode_predictions(scores, geometry1, geometry2
                                               )
    boxes, angles = non_max_suppression(np.array(bboxes), probs=confs, angles=np.array(angles))
    boxesangles = (boxes, angles)
    return boxesangles


def show_east(boxesangles, frame, **kwargs):
    bboxes = boxesangles[0]
    angles = boxesangles[1]
    for ((X0, Y0, X1, Y1), angle) in zip(bboxes, angles):
        width = abs(X0 - X1)
        height = abs(Y0 - Y1)
        cX = int(X0 + width * 0.5)
        cY = int(Y0 + height * 0.5)

        rotRect = ((cX, cY), ((X1 - X0), (Y1 - Y0)), angle * (-1))
        points = rotated_Rectangle(frame, rotRect, color=(255, 0, 0), thickness=1)
        cv2.polylines(frame, [points], isClosed=True, color=(255, 0, 0), thickness=1, lineType=cv2.LINE_8)

    return frame


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped

rclpy.init(args=None)

node = rclpy.create_node('minimal_publisher')

pub_img = node.create_publisher(Image, "/ocr_img", 10)
pub_text = node.create_publisher(String, "/ocr_text", 10)

pipeline = dai.Pipeline()
pipeline.setOpenVINOVersion(version=dai.OpenVINO.Version.VERSION_2021_2)

colorCam = pipeline.createColorCamera()
colorCam.setPreviewSize(256, 256)
colorCam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
colorCam.setInterleaved(False)
colorCam.setBoardSocket(dai.CameraBoardSocket.RGB)
colorCam.setFps(10)

controlIn = pipeline.createXLinkIn()
controlIn.setStreamName('control')
controlIn.out.link(colorCam.inputControl)


cam_xout = pipeline.createXLinkOut()
cam_xout.setStreamName("preview")

nn = pipeline.createNeuralNetwork()
model_filename='text-detection.blob'
nnPath = os.path.join(get_package_share_directory('blindbuy_oakd'), 'models', model_filename)
nn.setBlobPath(nnPath)
nn.setNumPoolFrames(1)
colorCam.preview.link(nn.input)
nn.passthrough.link(cam_xout.input)

nn_xout = pipeline.createXLinkOut()
nn_xout.setStreamName("detections")
nn.out.link(nn_xout.input)

manip = pipeline.createImageManip()
manip.setWaitForConfigInput(True)

manip_img = pipeline.createXLinkIn()
manip_img.setStreamName('manip_img')
manip_img.out.link(manip.inputImage)

manip_cfg = pipeline.createXLinkIn()
manip_cfg.setStreamName('manip_cfg')
manip_cfg.out.link(manip.inputConfig)

manip_xout = pipeline.createXLinkOut()
manip_xout.setStreamName('manip_out')

nn2 = pipeline.createNeuralNetwork()
model2_filename='text-recognition-0012.blob'
nn2Path = os.path.join(get_package_share_directory('blindbuy_oakd'), 'models', model2_filename)
nn2.setBlobPath(nn2Path)
nn2.setNumInferenceThreads(2)
manip.out.link(nn2.input)
manip.out.link(manip_xout.input)

nn2_xout = pipeline.createXLinkOut()
nn2_xout.setStreamName("recognitions")
nn2.out.link(nn2_xout.input)

device = dai.Device(pipeline)
device.startPipeline()

def to_tensor_result(packet):
    return {
        name: np.array(packet.getLayerFp16(name))
        for name in [tensor.name for tensor in packet.getRaw().tensors]
    }

q_prev = device.getOutputQueue("preview")
# This should be set to block, but would get to some extreme queuing/latency!
q_det = device.getOutputQueue("detections", 1, blocking=False)
q_rec = device.getOutputQueue("recognitions")

q_manip_img = device.getInputQueue("manip_img")
q_manip_cfg = device.getInputQueue("manip_cfg")
q_manip_out = device.getOutputQueue("manip_out")

controlQueue = device.getInputQueue('control')

frame = None
cropped_stacked = None
rotated_rectangles = []
rec_pushed = 0
rec_received = 0

class CTCCodec(object):
    """ Convert between text-label and text-index """
    def __init__(self, characters):
        # characters (str): set of the possible characters.
        dict_character = list(characters)

        self.dict = {}
        for i, char in enumerate(dict_character):
             self.dict[char] = i + 1

    
        self.characters = dict_character
        #print(self.characters)
        #input()
        
    def decode(self, preds):
        """ convert text-index into text-label. """
        texts = []
        index = 0
        # Select max probabilty (greedy decoding) then decode index to character
        preds = preds.astype(np.float16)
        preds_index = np.argmax(preds, 2)
        preds_index = preds_index.transpose(1, 0)
        preds_index_reshape = preds_index.reshape(-1)
        preds_sizes = np.array([preds_index.shape[1]] * preds_index.shape[0])

        for l in preds_sizes:
            t = preds_index_reshape[index:index + l]

            # NOTE: t might be zero size
            if t.shape[0] == 0:
                continue

            char_list = []
            for i in range(l):
                # removing repeated characters and blank.
                if not (i > 0 and t[i - 1] == t[i]):
                    if self.characters[t[i]] != '#':
                        char_list.append(self.characters[t[i]])
            text = ''.join(char_list)
            texts.append(text)

            index += l

        return texts

characters = '0123456789abcdefghijklmnopqrstuvwxyz#'
codec = CTCCodec(characters)

ctrl = dai.CameraControl()
ctrl.setAutoFocusMode(dai.CameraControl.AutoFocusMode.AUTO)
ctrl.setAutoFocusTrigger()
controlQueue.send(ctrl)

while True:
    in_prev = q_prev.tryGet()

    if in_prev is not None:
        shape = (3, in_prev.getHeight(), in_prev.getWidth())
        frame_orig = in_prev
        frame = in_prev.getData().reshape(shape).transpose(1, 2, 0).astype(np.uint8)
        frame = np.ascontiguousarray(frame)

    # Multiple recognition results may be available, read until queue is empty
    while True:
        in_rec = q_rec.tryGet()
        if in_rec is None:
            break
        rec_data = bboxes = np.array(in_rec.getFirstLayerFp16()).reshape(30,1,37)
        decoded_text = codec.decode(rec_data)[0]
        pos = rotated_rectangles[rec_received]
        print("{:2}: {:20}".format(rec_received, decoded_text),
              "center({:3},{:3}) size({:3},{:3}) angle{:5.1f} deg".format(
                  int(pos[0][0]), int(pos[0][1]), pos[1][0], pos[1][1], pos[2]))
        # Draw the text on the right side of 'cropped_stacked' - placeholder
        if cropped_stacked is not None:
            cv2.putText(cropped_stacked, decoded_text,
                            (120 + 10 , 32 * rec_received + 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            #cv2.imshow('cropped_stacked', cropped_stacked)
        rec_received += 1

        #Publish text to topic
        msg_text=String()
        msg_text.data=decoded_text
        pub_text.publish(msg_text) 
    if cv2.waitKey(1) == ord('q'):
        break

    if rec_received >= rec_pushed:
        in_det = q_det.tryGet()
        if in_det is not None:
            scores, geom1, geom2 = to_tensor_result(in_det).values()
            scores = np.reshape(scores, (1, 1, 64, 64))
            geom1 = np.reshape(geom1, (1, 4, 64, 64))
            geom2 = np.reshape(geom2, (1, 1, 64, 64))
    
            bboxes, confs, angles = decode_predictions(scores, geom1, geom2)
            boxes, angles = non_max_suppression(np.array(bboxes), probs=confs, angles=np.array(angles))
            rotated_rectangles = [
                get_cv_rotated_rect(bbox, angle * -1)
                for (bbox, angle) in zip(boxes, angles)
            ]

    if frame is not None:
        if in_det is not None:
            rec_received = 0
            rec_pushed = len(rotated_rectangles)
            if rec_pushed:
                print("====== Pushing for recognition, count:", rec_pushed)
            cropped_stacked = None
            for idx, rotated_rect in enumerate(rotated_rectangles):
                # Draw detection crop area on input frame
                points = np.int0(cv2.boxPoints(rotated_rect))
                cv2.polylines(frame, [points], isClosed=True, color=(255, 0, 0), thickness=1, lineType=cv2.LINE_8)

                # TODO make it work taking args like in OpenCV:
                # rr = ((256, 256), (128, 64), 30)
                rr = dai.RotatedRect()
                rr.center.x    = rotated_rect[0][0]
                rr.center.y    = rotated_rect[0][1]
                rr.size.width  = rotated_rect[1][0]
                rr.size.height = rotated_rect[1][1]
                rr.angle       = rotated_rect[2]
                cfg = dai.ImageManipConfig()
                cfg.setCropRotatedRect(rr, False)
                cfg.setResize(120, 32)
                # Send frame and config to device
                if idx == 0:
                    q_manip_img.send(frame_orig)
                else:
                    cfg.setReusePreviousImage(True)
                q_manip_cfg.send(cfg)
                # Get processed output from device
                cropped = q_manip_out.get()
                shape = (3, cropped.getHeight(), cropped.getWidth())
                transformed = cropped.getData().reshape(shape).transpose(1, 2, 0)

                rec_placeholder_img = np.zeros((32, 200, 3), np.uint8)
                transformed = np.hstack((transformed, rec_placeholder_img))
                if cropped_stacked is None:
                    cropped_stacked = transformed
                else:
                    cropped_stacked = np.vstack((cropped_stacked, transformed))
            # if cropped_stacked is not None:
            #     cv2.imshow('cropped_stacked', cropped_stacked)
            in_det = None


            #Publish img
            cvb = CvBridge()
            stamp = node.get_clock().now().to_msg()
            image_msg = cvb.cv2_to_imgmsg(frame, encoding='bgr8')
            image_msg.header.stamp = stamp
            image_msg.header.frame_id = 'oak-d_frame'
            
            pub_img.publish(image_msg)

            # cv2.imshow('preview', frame)
            key = cv2.waitKey(1)
            if  key == ord('q'):
                break
            elif key == ord('t'):
                print("Autofocus trigger (and disable continuous)")
                ctrl = dai.CameraControl()
                ctrl.setAutoFocusMode(dai.CameraControl.AutoFocusMode.AUTO)
                ctrl.setAutoFocusTrigger()
                controlQueue.send(ctrl)
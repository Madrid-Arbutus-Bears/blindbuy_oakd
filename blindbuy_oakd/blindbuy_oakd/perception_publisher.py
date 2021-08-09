#! /usr/bin/env python
# # coding=utf-8
import math
import cv2
import depthai as dai
import numpy as np
from imutils.video import FPS

import os
from ament_index_python.packages import get_package_share_directory

from pathlib import Path
from math import cos, sin
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from tf2_ros import TransformBroadcaster, TransformStamped

from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridge

class PalmDetection:
    def run_palm(self, frame, nn_data):
        """
        Each palm detection is a tensor consisting of 19 numbers:
            - ymin, xmin, ymax, xmax
            - x,y-coordinates for the 7 key_points
            - confidence score
        :return:
        """
        if nn_data is None:
            return
        shape = (128, 128)
        num_keypoints = 7
        min_score_thresh = 0.7
        anchors = np.load(os.path.join(get_package_share_directory('blindbuy_oakd'), "data", "array", "anchors_palm.npy"))

        # Run the neural network
        results = self.to_tensor_result(nn_data)

        raw_box_tensor = results.get("regressors").reshape(-1, 896, 18)  # regress
        raw_score_tensor = results.get("classificators").reshape(-1, 896, 1)  # classification

        detections = self.raw_to_detections(raw_box_tensor, raw_score_tensor, anchors, shape, num_keypoints)

        palm_coords = [
            self.frame_norm(frame, *obj[:4])
            for det in detections
            for obj in det
            if obj[-1] > min_score_thresh
        ]

        palm_confs = [
            obj[-1] for det in detections for obj in det if obj[-1] > min_score_thresh
        ]

        if len(palm_coords) == 0:
            return

        return self.non_max_suppression(
            boxes=np.concatenate(palm_coords).reshape(-1, 4),
            probs=palm_confs,
            overlapThresh=0.1,
        )

    def sigmoid(self, x):
        return (1.0 + np.tanh(0.5 * x)) * 0.5

    def decode_boxes(self, raw_boxes, anchors, shape, num_keypoints):
        """
        Converts the predictions into actual coordinates using the anchor boxes.
        Processes the entire batch at once.
        """
        boxes = np.zeros_like(raw_boxes)
        x_scale, y_scale = shape

        x_center = raw_boxes[..., 0] / x_scale * anchors[:, 2] + anchors[:, 0]
        y_center = raw_boxes[..., 1] / y_scale * anchors[:, 3] + anchors[:, 1]

        w = raw_boxes[..., 2] / x_scale * anchors[:, 2]
        h = raw_boxes[..., 3] / y_scale * anchors[:, 3]

        boxes[..., 1] = y_center - h / 2.0  # xmin
        boxes[..., 0] = x_center - w / 2.0  # ymin
        boxes[..., 3] = y_center + h / 2.0  # xmax
        boxes[..., 2] = x_center + w / 2.0  # ymax

        for k in range(num_keypoints):
            offset = 4 + k * 2
            keypoint_x = raw_boxes[..., offset] / x_scale * anchors[:, 2] + anchors[:, 0]
            keypoint_y = (
                raw_boxes[..., offset + 1] / y_scale * anchors[:, 3] + anchors[:, 1]
            )
            boxes[..., offset] = keypoint_x
            boxes[..., offset + 1] = keypoint_y

        return boxes

    def raw_to_detections(self, raw_box_tensor, raw_score_tensor, anchors_, shape, num_keypoints):
        """

        This function converts these two "raw" tensors into proper detections.
        Returns a list of (num_detections, 17) tensors, one for each image in
        the batch.

        This is based on the source code from:
        mediapipe/calculators/tflite/tflite_tensors_to_detections_calculator.cc
        mediapipe/calculators/tflite/tflite_tensors_to_detections_calculator.proto
        """
        detection_boxes = self.decode_boxes(raw_box_tensor, anchors_, shape, num_keypoints)
        detection_scores = self.sigmoid(raw_score_tensor).squeeze(-1)
        output_detections = []
        for i in range(raw_box_tensor.shape[0]):
            boxes = detection_boxes[i]
            scores = np.expand_dims(detection_scores[i], -1)
            output_detections.append(np.concatenate((boxes, scores), -1))
        return output_detections

    def non_max_suppression(self, boxes, probs=None, angles=None, overlapThresh=0.3):
        if len(boxes) == 0:
            return [], []

        if boxes.dtype.kind == "i":
            boxes = boxes.astype("float")

        pick = []

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = y2

        if probs is not None:
            idxs = probs

        idxs = np.argsort(idxs)

        while len(idxs) > 0:
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)

            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])

            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)

            overlap = (w * h) / area[idxs[:last]]

            idxs = np.delete(
                idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0]))
            )

        if angles is not None:
            return boxes[pick].astype("int"), angles[pick]
        return boxes[pick].astype("int")

    def to_tensor_result(self, packet):
        return {
            name: np.array(packet.getLayerFp16(name))
            for name in [tensor.name for tensor in packet.getRaw().tensors]
        }

    def frame_norm(self, frame, *xy_vals):
        """
        nn data, being the bounding box locations, are in <0..1> range -
        they need to be normalized with frame width/height

        :param frame:
        :param xy_vals: the bounding box locations
        :return:
        """
        return (
            np.clip(np.array(xy_vals), 0, 1)
            * np.array(frame.shape[:2] * (len(xy_vals) // 2))[::-1]
        ).astype(int)


DEPTH_THRESH_HIGH = 5000
DEPTH_THRESH_LOW = 300
CAM_FPS = 30
RED_RATIO_PALM = 0.37
RED_RATIO_FACE = 0.2

# Required information for calculating spatial coordinates on the host
mono_HFOV = np.deg2rad(73.5)
depth_Width = 1080.0

def calc_angle(monoHFOV, depthWidth, offset):
    return math.atan(math.tan(monoHFOV / 2.0) * offset / (depthWidth / 2.0))

def crop_to_rect(frame):
    height = frame.shape[0]
    width  = frame.shape[1]
    delta = int((width-height) / 2)
    # print(height, width, delta)
    return frame[0:height, delta:width-delta]


# Calculate spatial coordinates from depth map and bounding box (ROI)
def calc_spatials(bbox, depth,ratio,filter="mean"):
    # Decrese the ROI to 1/3 of the original ROI
    deltaX = int((bbox[2] - bbox[0]) * ratio)
    deltaY = int((bbox[3] - bbox[1]) * ratio)
    bbox[0] = bbox[0] + deltaX
    bbox[1] = bbox[1] + deltaY
    bbox[2] = bbox[2] - deltaX
    bbox[3] = bbox[3] - deltaY

    # Calculate the average depth in the ROI. TODO: median, median /w bins, mode
    cnt = 0.0
    sum = 0.0
    median = []
    for x in range(bbox[2] - bbox[0]):
        for y in range(bbox[3] - bbox[1]):
            depthPixel = depth[bbox[1] + y][bbox[0] + x]
            if DEPTH_THRESH_LOW < depthPixel and depthPixel < DEPTH_THRESH_HIGH:
                cnt+=1.0
                sum+=depthPixel
                median.append(depthPixel)

    medianDepth = np.median(np.array(median))
    averageDepth = sum / cnt if 0 < cnt else 0

    # Detection centroid
    centroidX = int((bbox[2] - bbox[0]) / 2) + bbox[0]
    centroidY = int((bbox[3] - bbox[1]) / 2) + bbox[1]

    mid = int(depth.shape[0] / 2) # middle of the depth img
    bb_x_pos = centroidX - mid
    bb_y_pos = centroidY - mid

    angle_x = calc_angle(mono_HFOV, depth_Width, bb_x_pos)
    angle_y = calc_angle(mono_HFOV, depth_Width, bb_y_pos)

    if filter == "median":
        z = medianDepth
    elif filter == "mean":
        z = averageDepth
    else:
        z = averageDepth

    x = z * math.tan(angle_x)
    y = -z * math.tan(angle_y)

    # print(f"X: {x}mm, Y: {y} mm, Z: {z} mm")
    return (x,y,z, centroidX, centroidY)

def draw_bbox(debug_frame, bbox, color):
    def draw(img):
        cv2.rectangle(
            img=img,
            pt1=(bbox[0], bbox[1]),
            pt2=(bbox[2], bbox[3]),
            color=color,
            thickness=2,
        )
    draw(debug_frame)

def draw_man(debug_frame, bbox, c_cords, color):

    cv2.line(debug_frame, c_cords, (bbox[0], bbox[1]), color, 2)
    cv2.line(debug_frame, c_cords, (bbox[2], bbox[3] - (bbox[3] - bbox[1])), color, 2)
    draw_x = [bbox[0],bbox[0] + int((bbox[2] - bbox[0])/2), bbox[2]]
    draw_y = [bbox[1],bbox[1]-12, bbox[3] -(bbox[3] - bbox[1])]
    draw_points = (np.asarray([draw_x, draw_y]).T).astype(np.int32)
    cv2.polylines(debug_frame, [draw_points], False, color, 2)
 
def draw_palm_detection(debug_frame, palm_coords, depth):
    if palm_coords is None: return None

    color = (75, 25, 227)
    for bbox in palm_coords:
        draw_bbox(debug_frame,bbox, color)
        bbox_og = bbox.copy()
        spatialCoords = calc_spatials(bbox, depth,RED_RATIO_PALM)
        x,y,z,cx,cy = spatialCoords
        #draw_man(debug_frame, bbox_og, (cx,cy), color)
        #print("{0},{1},{2},{3},{4}".format(x,y,z,cx,cy))
        cv2.rectangle(debug_frame, (bbox_og[2]-75, bbox_og[3]), (bbox_og[2]+1, bbox_og[3]+35), color, -1)
        cv2.putText(debug_frame, f"X: {int(x)} mm", (bbox_og[2]-70, bbox_og[3]+10), cv2.FONT_HERSHEY_DUPLEX, 0.3, (175,175,175))
        cv2.putText(debug_frame, f"Y: {int(y)} mm", (bbox_og[2]-70, bbox_og[3]+20), cv2.FONT_HERSHEY_DUPLEX, 0.3, (175,175,175))
        cv2.putText(debug_frame, f"Z: {int(z)} mm", (bbox_og[2]-70, bbox_og[3]+30), cv2.FONT_HERSHEY_DUPLEX, 0.3, (175,175,175))

        return spatialCoords

def create_advanced_pipeline():
    print("Creating pipeline...")
    pipeline = dai.Pipeline()

    cam = pipeline.createColorCamera()
    cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam.setIspScale(2, 3) # To match 720P mono cameras
    cam.setBoardSocket(dai.CameraBoardSocket.RGB)
    cam.initialControl.setManualFocus(130)
    
    cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    cam.setPreviewSize(300, 300)
    cam.setFps(CAM_FPS)
    cam.setInterleaved(False)

    isp_xout = pipeline.createXLinkOut()
    isp_xout.setStreamName("cam")
    cam.isp.link(isp_xout.input)

    print(f"Creating palm detection Neural Network...")
    model_nn = pipeline.createNeuralNetwork()
    model_nn.setBlobPath(os.path.join(get_package_share_directory('blindbuy_oakd'), 'models', "palm_detection_openvino_2021.3_6shave.blob"))
    model_nn.input.setBlocking(False)

    # For Palm-detection NN
    manip = pipeline.createImageManip()
    manip.initialConfig.setResize(128, 128)
    cam.preview.link(manip.inputImage)
    manip.out.link(model_nn.input)

    model_nn_xout = pipeline.createXLinkOut()
    model_nn_xout.setStreamName("palm_nn")
    model_nn.out.link(model_nn_xout.input)

    # Creating left/right mono cameras for StereoDepth
    left = pipeline.createMonoCamera()
    left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
    left.setBoardSocket(dai.CameraBoardSocket.LEFT)

    right = pipeline.createMonoCamera()
    right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
    right.setBoardSocket(dai.CameraBoardSocket.RIGHT)

    # Create StereoDepth node that will produce the depth map
    stereo = pipeline.createStereoDepth()
    stereo.initialConfig.setConfidenceThreshold(245)
    stereo.initialConfig.setMedianFilter(dai.StereoDepthProperties.MedianFilter.KERNEL_7x7)
    stereo.setLeftRightCheck(True)
    stereo.setDepthAlign(dai.CameraBoardSocket.RGB)
    left.out.link(stereo.left)
    right.out.link(stereo.right)

    depth_out = pipeline.createXLinkOut()
    depth_out.setStreamName("depth")
 
    stereo.depth.link(depth_out.input)

    # NeuralNetwork
    print("Creating Face Detection Neural Network...")
    face_nn = pipeline.createNeuralNetwork()
    face_nn.setBlobPath(os.path.join(get_package_share_directory('blindbuy_oakd'), 'models', "face-detection-retail-0004_openvino_2021.2_4shave.blob"))

    cam.preview.link(face_nn.input)
 
    face_nn_xout = pipeline.createXLinkOut()
    face_nn_xout.setStreamName("face_nn")
    face_nn.out.link(face_nn_xout.input)

    # NeuralNetwork
    print("Creating Head Pose Neural Network...")
    pose_nn = pipeline.createNeuralNetwork()
    pose_nn.setBlobPath(os.path.join(get_package_share_directory('blindbuy_oakd'), 'models', "head-pose-estimation-adas-0001_openvino_2021.2_4shave.blob"))
    pose_nn_xin = pipeline.createXLinkIn()
    pose_nn_xin.setStreamName("pose_in")
    pose_nn_xin.out.link(pose_nn.input)
    pose_nn_xout = pipeline.createXLinkOut()
    pose_nn_xout.setStreamName("pose_nn")
    pose_nn.out.link(pose_nn_xout.input)

    print("Pipeline created.")
    return pipeline

def show_depth(depthFrame):
    depthFrameColor = cv2.normalize(depthFrame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
    depthFrameColor = cv2.equalizeHist(depthFrameColor)
    depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_JET)
    cv2.imshow("depth",depthFrameColor)

def bbox_face_extraction(in_face):
    bboxes = np.array(in_face.getFirstLayerFp16())
    bboxes = bboxes.reshape((bboxes.size // 7, 7))
    bboxes = bboxes[bboxes[:, 2] > 0.7][:, 3:7]

    return bboxes

def to_planar(arr: np.ndarray, shape: tuple) -> list:
    return [val for channel in cv2.resize(arr, shape).transpose(2, 0, 1) for y_col in channel for val in y_col]

def frame_norm(frame, bbox):
    norm_vals = np.full(len(bbox), frame.shape[0])
    norm_vals[::2] = frame.shape[1]
    return (np.clip(np.array(bbox), 0, 1) * norm_vals).astype(int)

def to_tensor_result(packet):
    return {
        tensor.name: np.array(packet.getLayerFp16(tensor.name)).reshape(tensor.dims)
        for tensor in packet.getRaw().tensors
    }

def draw_3d_axis(image, head_pose, origin, size=50):
    roll = head_pose[0] * np.pi / 180
    pitch = head_pose[1] * np.pi / 180
    yaw = -(head_pose[2] * np.pi / 180)

    # X axis (red)
    x1 = size * (cos(yaw) * cos(roll)) + origin[0]
    y1 = size * (cos(pitch) * sin(roll) + cos(roll) * sin(pitch) * sin(yaw)) + origin[1]
    cv2.line(image, (origin[0], origin[1]), (int(x1), int(y1)), (0, 0, 255), 3)

    # Y axis (green)
    x2 = size * (-cos(yaw) * sin(roll)) + origin[0]
    y2 = size * (-cos(pitch) * cos(roll) - sin(pitch) * sin(yaw) * sin(roll)) + origin[1]
    cv2.line(image, (origin[0], origin[1]), (int(x2), int(y2)), (0, 255, 0), 3)

    # Z axis (blue)
    x3 = size * (-sin(yaw)) + origin[0]
    y3 = size * (cos(yaw) * sin(pitch)) + origin[1]
    cv2.line(image, (origin[0], origin[1]), (int(x3), int(y3)), (255, 0, 0), 2)

def draw_pose_data(debug_frame, head_pose, origin,color):
    cv2.rectangle(debug_frame, (origin[3]-75, origin[4]), (origin[3]+1, origin[4]+55), color, -1)
    cv2.putText(debug_frame, f"X: {int(origin[0])} mm", (origin[3]-70, origin[4]+10), cv2.FONT_HERSHEY_DUPLEX, 0.3, (175,175,175))
    cv2.putText(debug_frame, f"Y: {int(origin[1])} mm", (origin[3]-70, origin[4]+20), cv2.FONT_HERSHEY_DUPLEX, 0.3, (175,175,175))
    cv2.putText(debug_frame, f"Z: {int(origin[2])} mm", (origin[3]-70, origin[4]+30), cv2.FONT_HERSHEY_DUPLEX, 0.3, (175,175,175))
    cv2.putText(debug_frame, f"r: {int(head_pose[0])} º", (origin[3]-70, origin[4]+30), cv2.FONT_HERSHEY_DUPLEX, 0.3, (175,175,175))
    cv2.putText(debug_frame, f"p: {int(head_pose[1])} º", (origin[3]-70, origin[4]+40), cv2.FONT_HERSHEY_DUPLEX, 0.3, (175,175,175))
    cv2.putText(debug_frame, f"w: {int(head_pose[2])} º", (origin[3]-70, origin[4]+50), cv2.FONT_HERSHEY_DUPLEX, 0.3, (175,175,175))


class PerceptionPublisher(Node):

    def __init__(self):
        rclpy.init()
        super().__init__('oakd_publisher')
        qos_profile = QoSProfile(depth=10)
        self.broadcaster = TransformBroadcaster(self, qos=qos_profile)
        
        #Image publisher
        #self.pub_depth_img = self.create_publisher(sensor_msgs.msg.Image, "/depth_img", 10)
        self.pub_rectified_img = self.create_publisher(Image, "/rectified_img", 10)
        
        # Transform for head declaration
        self.transform_head = TransformStamped()
        self.transform_head.header.frame_id = 'oak-d_camera_center'
        self.transform_head.child_frame_id = 'head'

        # Transform for palm declaration
        self.transform_palm = TransformStamped()
        self.transform_palm.header.frame_id = 'oak-d_camera_center'
        self.transform_palm.child_frame_id = 'palm'
        #fixed Identity quaternion (no rotation)

        self.fps = FPS()
        self.advanced_main()

    def pose_to_quaternion(self,head_pose):
        # roll = head_pose[1] * np.pi / 180
        # pitch = head_pose[2] * np.pi / 180
        # yaw = head_pose[0] * np.pi / 180
        roll = -(head_pose[0] * np.pi / 180)
        pitch = head_pose[2] * np.pi / 180
        yaw = head_pose[1] * np.pi / 180

        qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
        qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
        qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
        qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)

        return [qx, qy, qz, qw]

    def publish_palm_transform(self, translation):
        # Update transform
        now = self.get_clock().now()
        self.transform_palm.header.stamp = now.to_msg()
        self.transform_palm.transform.translation.x = -translation[0]*0.001
        self.transform_palm.transform.translation.y = translation[1]*0.001
        self.transform_palm.transform.translation.z = translation[2]*0.001

        # Send transform
        self.broadcaster.sendTransform(self.transform_palm)

    def publish_head_transform(self,head_loc, euler_ang):
        # Update transform
        now = self.get_clock().now()
        self.transform_head.header.stamp = now.to_msg()
        self.transform_head.transform.translation.x = -head_loc[0]*0.001
        self.transform_head.transform.translation.y = head_loc[1]*0.001
        self.transform_head.transform.translation.z = head_loc[2]*0.001
        
        rotation=self.pose_to_quaternion(euler_ang)
        self.transform_head.transform.rotation.x=rotation[0]
        self.transform_head.transform.rotation.y=rotation[1]
        self.transform_head.transform.rotation.z=rotation[2]
        self.transform_head.transform.rotation.w=rotation[3]

        # Send transform
        self.broadcaster.sendTransform(self.transform_head)

    def advanced_main(self):

        pipeline = create_advanced_pipeline()
        with dai.Device(pipeline) as device:
            # Create output queues
            vidQ = device.getOutputQueue(name="cam", maxSize=1, blocking=False)
            depthQ = device.getOutputQueue(name="depth", maxSize=1, blocking=False)
            palmQ = device.getOutputQueue(name="palm_nn", maxSize=1, blocking=False)
            faceQ = device.getOutputQueue("face_nn",maxSize=1, blocking=False)
            pose_inQ = device.getInputQueue("pose_in",maxSize=1, blocking=False)
            pose_outQ = device.getOutputQueue(name="pose_nn", maxSize=1, blocking=False)

            palmDetection = PalmDetection()

            depthFrame = None
            frame = None
            head_loc = None

            print("Main loop init")

            self.fps.start()

            while rclpy.ok():

                in_rgb = vidQ.tryGet()
                if in_rgb is not None:
                    frame = crop_to_rect(in_rgb.getCvFrame())
                    debug_frame = frame.copy()

                in_depth = depthQ.tryGet()
                if in_depth is not None:
                    depthFrame = crop_to_rect(in_depth.getFrame())


                in_face = faceQ.tryGet()
                if in_face is not None and frame is not None and depthFrame is not None:
                    
                    bboxes = bbox_face_extraction(in_face)
                    color=(143, 184, 77)
                    for raw_bbox in bboxes:

                        bbox = frame_norm(frame, raw_bbox)
                        det_frame = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]

                        pose_data = dai.NNData()
                        pose_data.setLayer("data", to_planar(det_frame, (60, 60)))
                        pose_inQ.send(pose_data)

                        draw_bbox(debug_frame,bbox,color)
                        head_loc = calc_spatials(bbox,depthFrame,RED_RATIO_FACE,filter="median")

                palm_in = palmQ.tryGet()

                if palm_in is not None and frame is not None and depthFrame is not None:

                    #perform computation and output drawing
                    palm_coords = palmDetection.run_palm(debug_frame, palm_in)
                    # Calculate and draw spatial coordinates of the palm
                    spatialCoords = draw_palm_detection(debug_frame, palm_coords, depthFrame)

                    #publish palm transform
                    if spatialCoords is not None:
                        self.publish_palm_transform(spatialCoords)

                    #publish detection image
                    cvb = CvBridge()
                    stamp = self.get_clock().now().to_msg()
                    image_msg = cvb.cv2_to_imgmsg(debug_frame, encoding='bgr8')
                    image_msg.header.stamp = stamp
                    image_msg.header.frame_id = 'oak-d_frame'
                    
                    self.pub_rectified_img.publish(image_msg)

                    ###### IMSHOW FOR DEPTH AND FRAME
                    #cv2.imshow("debug", debug_frame)
                    #show_depth(depthFrame)

                head_or = pose_outQ.tryGet()

                if head_or is not None:
                    pose = [val[0][0] for val in to_tensor_result(head_or).values()]
                    if head_loc[2] is not np.nan:
                        self.publish_head_transform(head_loc,pose)
                        print("Loc:({0},{1},{2}) , Or: ({3},{4},{5})".format(head_loc[0],head_loc[1],head_loc[2],pose[0],pose[1],pose[2]))
                    #draw_3d_axis(debug_frame,pose,(head_pose[3],head_pose[4]),100)
                    #draw_pose_data(debug_frame,pose,head_pose,color=(143, 184, 77))

                self.fps.update()

                if cv2.waitKey(1) == ord('q'):
                    cv2.destroyAllWindows()
                    self.fps.stop()
                    print("CAM FPS: {0}  P-FPS:{1}".format(CAM_FPS,self.fps.fps()))
                    self.destroy_node()

def main():
    node = PerceptionPublisher()


if __name__ == '__main__':
    main()

#! /usr/bin/env python

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from tf2_ros import TransformBroadcaster, TransformStamped

from ament_index_python.packages import get_package_share_directory
import os
import cv2
import depthai as dai
import numpy as np
import time

from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridge

class FaceDetection(Node):

    def __init__(self):
        rclpy.init()
        super().__init__('face_detection')
        qos_profile = QoSProfile(depth=10)
        self.broadcaster = TransformBroadcaster(self, qos=qos_profile)
        self.pub_point = self.create_publisher(PointStamped, "/head_detection", 10)
        #Image publisher
        #self.pub_depth_img = self.create_publisher(sensor_msgs.msg.Image, "/depth_img", 10)
        self.pub_rectified_img = self.create_publisher(Image, "/rectified_img", 10)
        
        # Transform declaration
        self.transform = TransformStamped()
        self.transform.header.frame_id = 'oak-d_camera_center'
        self.transform.child_frame_id = 'head'

        # Detect face
        syncNN = True
        self.flipRectified = True

        # Get argument first
        model_filename='face-detection-retail-0004_openvino_2021.2_4shave.blob'
        nnPath = os.path.join(get_package_share_directory('blindbuy_oakd'), 'models', model_filename)
        print(nnPath)

        # Start defining a pipeline
        self.pipeline = dai.Pipeline()

        manip = self.pipeline.createImageManip()
        manip.initialConfig.setResize(300, 300)
        # The NN model expects BGR input. By default ImageManip output type would be same as input (gray in this case)
        manip.initialConfig.setFrameType(dai.RawImgFrame.Type.BGR888p)
        manip.setKeepAspectRatio(False) #Squeeze image without cropping to fit 300x300 nn input

        # Define a neural network that will make predictions based on the source frames
        spatialDetectionNetwork = self.pipeline.createMobileNetSpatialDetectionNetwork()
        spatialDetectionNetwork.setConfidenceThreshold(0.5)
       
        spatialDetectionNetwork.setBlobPath(nnPath)
        spatialDetectionNetwork.input.setBlocking(False)
        spatialDetectionNetwork.setBoundingBoxScaleFactor(0.5)
        spatialDetectionNetwork.setDepthLowerThreshold(100)
        spatialDetectionNetwork.setDepthUpperThreshold(5000)

        manip.out.link(spatialDetectionNetwork.input)

        # Create outputs
        xoutManip = self.pipeline.createXLinkOut()
        xoutManip.setStreamName("right")
        if(syncNN):
            spatialDetectionNetwork.passthrough.link(xoutManip.input)
        else:
            manip.out.link(xoutManip.input)

        depthRoiMap = self.pipeline.createXLinkOut()
        depthRoiMap.setStreamName("boundingBoxDepthMapping")

        xoutDepth = self.pipeline.createXLinkOut()
        xoutDepth.setStreamName("depth")

        nnOut = self.pipeline.createXLinkOut()
        nnOut.setStreamName("detections")
        spatialDetectionNetwork.out.link(nnOut.input)
        spatialDetectionNetwork.boundingBoxMapping.link(depthRoiMap.input)

        monoLeft = self.pipeline.createMonoCamera()
        monoRight = self.pipeline.createMonoCamera()
        stereo = self.pipeline.createStereoDepth()
        monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
        monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)
        stereo.setOutputDepth(True)
        stereo.setConfidenceThreshold(255)
        stereo.setOutputRectified(True)

        stereo.rectifiedRight.link(manip.inputImage)

        monoLeft.out.link(stereo.left)
        monoRight.out.link(stereo.right)

        stereo.depth.link(spatialDetectionNetwork.inputDepth)
        spatialDetectionNetwork.passthroughDepth.link(xoutDepth.input)

        self.detect_face()

    def detect_face(self):
        # Pipeline defined, now the device is connected to
        with dai.Device(self.pipeline) as device:
            # Start pipeline
            device.startPipeline()

            # Output queues will be used to get the rgb frames and nn data from the outputs defined above
            previewQueue = device.getOutputQueue(name="right", maxSize=4, blocking=False)
            detectionNNQueue = device.getOutputQueue(name="detections", maxSize=4, blocking=False)
            depthRoiMap = device.getOutputQueue(name="boundingBoxDepthMapping", maxSize=4, blocking=False)
            depthQueue = device.getOutputQueue(name="depth", maxSize=4, blocking=False)

            rectifiedRight = None
            detections = []

            startTime = time.monotonic()
            counter = 0
            fps = 0
            color = (255, 255, 255)

            while rclpy.ok():

                inRectified = previewQueue.get()
                det = detectionNNQueue.get()
                depth = depthQueue.get()

                counter += 1
                currentTime = time.monotonic()
                if (currentTime - startTime) > 1:
                    fps = counter / (currentTime - startTime)
                    counter = 0
                    startTime = currentTime

                rectifiedRight = inRectified.getCvFrame()

                depthFrame = depth.getFrame()

                depthFrameColor = cv2.normalize(depthFrame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
                depthFrameColor = cv2.equalizeHist(depthFrameColor)
                depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_HOT)
                detections = det.detections
                if len(detections) != 0:
                    boundingBoxMapping = depthRoiMap.get()
                    roiDatas = boundingBoxMapping.getConfigData()

                    for roiData in roiDatas:
                        roi = roiData.roi
                        roi = roi.denormalize(depthFrameColor.shape[1], depthFrameColor.shape[0])
                        topLeft = roi.topLeft()
                        bottomRight = roi.bottomRight()
                        xmin = int(topLeft.x)
                        ymin = int(topLeft.y)
                        xmax = int(bottomRight.x)
                        ymax = int(bottomRight.y)
                        cv2.rectangle(depthFrameColor, (xmin, ymin), (xmax, ymax), color, cv2.FONT_HERSHEY_SCRIPT_SIMPLEX)

                if self.flipRectified:
                    rectifiedRight = cv2.flip(rectifiedRight, 1)

                # if the rectifiedRight is available, draw bounding boxes on it and show the rectifiedRight
                height = rectifiedRight.shape[0]
                width = rectifiedRight.shape[1]
                for detection in detections:
                    if self.flipRectified:
                        swap = detection.xmin
                        detection.xmin = 1 - detection.xmax
                        detection.xmax = 1 - swap
                    # denormalize bounding box
                    x1 = int(detection.xmin * width)
                    x2 = int(detection.xmax * width)
                    y1 = int(detection.ymin * height)
                    y2 = int(detection.ymax * height)

                    x=detection.spatialCoordinates.x
                    y=detection.spatialCoordinates.y
                    z=detection.spatialCoordinates.z

                    cv2.putText(rectifiedRight, "{:.2f}".format(detection.confidence*100), (x1 + 10, y1 + 35), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                    cv2.putText(rectifiedRight, f"X: {int(x)} mm", (x1 + 10, y1 + 50), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                    cv2.putText(rectifiedRight, f"Y: {int(y)} mm", (x1 + 10, y1 + 65), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                    cv2.putText(rectifiedRight, f"Z: {int(z)} mm", (x1 + 10, y1 + 80), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)

                    #Publish Transform
                    self.publish_transform((float(-x)/1000,float(y)/1000,float(z)/1000)) #Publish transform in meters

                    #Publish PointStamped
                    point_msg=PointStamped()
                    point_msg.point.x=float(x)
                    point_msg.point.y=float(y)
                    point_msg.point.z=float(z)
                    stamp = self.get_clock().now().to_msg()
                    point_msg.header.stamp = stamp
                    point_msg.header.frame_id = 'oak-d_frame'
                
                    self.pub_point.publish(point_msg)

                    cv2.rectangle(rectifiedRight, (x1, y1), (x2, y2), color, cv2.FONT_HERSHEY_SIMPLEX)

                cvb = CvBridge()
                cv2.putText(rectifiedRight, "NN fps: {:.2f}".format(fps), (2, rectifiedRight.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, color)
                #cv2.imshow("depth", depthFrameColor)
                #self.pub_depth_img.publish(cvb.cv2_to_imgmsg(depthFrameColor))
                #cv2.imshow("rectified right", rectifiedRight)
                frame=rectifiedRight
                stamp = self.get_clock().now().to_msg()
                image_msg = cvb.cv2_to_imgmsg(frame, encoding='bgr8')
                image_msg.header.stamp = stamp
                image_msg.header.frame_id = 'oak-d_frame'
                
                self.pub_rectified_img.publish(image_msg)

                if cv2.waitKey(1) == ord('q'):
                    break

    def publish_transform(self, translation):
        # Update transform
        now = self.get_clock().now()
        self.transform.header.stamp = now.to_msg()
        # self.transform.transform.translation.y = translation[0]
        # self.transform.transform.translation.z = translation[1]
        # self.transform.transform.translation.x = translation[2]
        self.transform.transform.translation.x = translation[0]
        self.transform.transform.translation.y = translation[1]
        self.transform.transform.translation.z = translation[2]

        # Send transform
        self.broadcaster.sendTransform(self.transform)

def main():
    node = FaceDetection()


if __name__ == '__main__':
    main()

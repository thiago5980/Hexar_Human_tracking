#!/usr/bin/env python3.6
import pyrealsense2 as rs
# Import Numpy for easy array manipulation
import numpy as np
# Import OpenCV for easy image rendering
import cv2

# Create a pipeline
pipeline = rs.pipeline()

# Create a config and configure the pipeline to stream
#  different resolutions of color and depth streams
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

if device_product_line == 'L500':
    config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
else:
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
profile = pipeline.start(config)

# Getting the depth sensor's depth scale (see rs-align example for explanation)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale is: " , depth_scale)

# We will be removing the background of objects more than
#  clipping_distance_in_meters meters away
clipping_distance_in_meters = 1 #1 meter
clipping_distance = clipping_distance_in_meters / depth_scale

# Create an align object
# rs.align allows us to perform alignment of depth frames to others frames
# The "align_to" is the stream type to which we plan to align depth frames.
align_to = rs.stream.color
align = rs.align(align_to)




import argparse
import rospy 
import gi
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
gi.require_version('Gtk', '2.0')
from std_msgs.msg import Float64MultiArray  
import os
# limit the number of cpus used by high performance libraries
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import numpy as np
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn
from PIL import Image, ImageDraw

import logging
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.dataloaders import VID_FORMATS, LoadImages, LoadStreams
from yolov5.utils.general import (LOGGER, check_img_size, non_max_suppression, scale_coords, check_requirements, cv2,
                                  check_imshow, xyxy2xywh, increment_path, strip_optimizer, colorstr, print_args, check_file)
from yolov5.utils.torch_utils import select_device, time_sync
from yolov5.utils.plots import Annotator, colors, save_one_box
from trackers.multi_tracker_zoo import create_tracker
from yolov5.utils.augmentations import letterbox
# remove duplicated stream handler to avoid duplicated logging
logging.getLogger().removeHandler(logging.getLogger().handlers[0])

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov5 strongsort root directory
WEIGHTS = ROOT / 'weights'

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if str(ROOT / 'yolov5') not in sys.path:
    sys.path.append(str(ROOT / 'yolov5'))  # add yolov5 ROOT to PATH
if str(ROOT / 'strong_sort') not in sys.path:
    sys.path.append(str(ROOT / 'strong_sort'))  # add strong_sort ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


cnt = 0
f_track = 0
d_cnt = 0
track_miss = False
track_true = False

@torch.no_grad()

class track_person:
    def __init__(self):
        self.source = '0'
        self.yolo_weights = '/home/sol/Yolov5_StrongSORT_OSNet/yolov5n.pt'
        self.reid_weights = WEIGHTS / 'osnet_x0_25_msmt17.pt'
        # self.config_strongsort = '/home/jetson/catkin_ws/src/track/scripts/strong_sort/configs/strong_sort.yaml'
        self.tracking_method='strongsort'
        self.imgsz = (640, 640)
        self.conf_thres = 0.25
        self.iou_thres = 0.45
        self.max_det = 1000
        self.device = '0'
        self.show_vid=False,  # show results
        self.save_txt=False,  # save results to *.txt
        self.save_conf=False,  # save confidences in --save-txt labels
        self.save_crop=False,  # save cropped prediction boxes
        self.save_vid=False,  # save confidences in --save-txt labels
        self.nosave=False,  # do not save images/videos
        self.classes=None,  # filter by class: --class 0, or --class 0 2 3
        self.agnostic_nms=False,  # class-agnostic NMS
        self.augment=False,  # augmented inference
        self.visualize=False,  # visualize features
        self.update=False,  # update all models
        self.project='/home/airo/Yolov5_StrongSORT_OSNet/runs/track',  # save results to project/name
        self.name='exp',  # save results to project/name
        self.exist_ok=False,  # existing project/name ok, do not increment
        self.line_thickness=3,  # bounding box thickness (pixels)
        self.hide_labels=False,  # hide labels
        self.hide_conf=False,  # hide confidences
        self.hide_class=False,  # hide IDs
        self.half=False,  # use FP16 half-precision inference
        self.dnn=False,  # use OpenCV DNN for ONNX inference
        self.bridge = CvBridge()
        device = select_device(self.device)
        self.model = DetectMultiBackend(self.yolo_weights, device = device, dnn = self.dnn, data = None, fp16=False)
        self.stride, selfnames, self.pt = self.model.stride, self.model.names, self.model.pt
        self.sub = rospy.Subscriber("/camera/aligned_depth_to_color/image_raw", Image, self.rgb_callback)




    # Img.data.ima
    def rgb_callback(self, data):
        # try:
        #     img = self.bridge.imgmsg_to_cv2(data, "bgr8")
        # except CvBridgeError as e:
        #     rospy.logerr(e)
        # rospy.loginfo("rgb callback")

        frames = pipeline.wait_for_frames()
        # frames.get_depth_frame() is a 640x360 depth image

        # Align the depth frame to color frame
        aligned_frames = align.process(frames)

        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
        color_frame = aligned_frames.get_color_frame()

        # Validate that both frames are valid
        # if not aligned_depth_frame or not color_frame:
        #     continue
        cv2.imshow("test", color_frame)
        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        img = color_image.copy()
        dt, seen = [0.0, 0.0, 0.0, 0.0], 0
        t1 = time_sync()
        im = letterbox(img, self.imgsz, stride=self.stride, auto=self.pt)[0]
        im = im.transpose((2, 0, 1))[::-1]
        print(im.shape)
        im = np.ascontiguousarray(im)
        im = torch.from_numpy(im).to(self.device)
        im = im.half() if self.model.fp16 else im.float()
        im /= 255

        # self.cfg = get_config()
        # self.cfg.merge_from_file(self.config_strongsort)
        nr_sources = 1
        curr_frames, prev_frames = [None] * nr_sources, [None] * nr_sources
        # strongsort_list = []
        # for i in range(nr_sources):
        #     strongsort_list.append(
        #         StrongSORT(
        #             self.strong_sort_weights,
        #             self.device,
        #             max_dist=self.cfg.STRONGSORT.MAX_DIST,
        #             max_iou_distance=self.cfg.STRONGSORT.MAX_IOU_DISTANCE,
        #             max_age=self.cfg.STRONGSORT.MAX_AGE,
        #             n_init=self.cfg.STRONGSORT.N_INIT,
        #             nn_budget=self.cfg.STRONGSORT.NN_BUDGET,
        #             mc_lambda=self.cfg.STRONGSORT.MC_LAMBDA,
        #             ema_alpha=self.cfg.STRONGSORT.EMA_ALPHA,

        #         )
        #     )
        # outputs = [None] * nr_sources
        tracker_list = []
        for i in range(nr_sources):
            tracker = create_tracker(self.tracking_method, self.reid_weights, self.device, self.half)
            tracker_list.append(tracker, )
            if hasattr(tracker_list[i], 'model'):
                if hasattr(tracker_list[i].model, 'warmup'):
                    tracker_list[i].model.warmup()
        outputs = [None] * nr_sources
        if len(im.shape) == 3:
            im = im[None]   
        t2 = time_sync()
        dt[0] += t2 - t1

        pred = self.model(im, augment=False, visualize=False)
        t3 = time_sync()
        dt[1] += t3 - t2
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, max_det=self.max_det)
        dt[2] += time_sync() - t3

        for i, det in enumerate(pred):
            seen += 1
            testt = []
            testd = []
            img_P = Image.fromarray(img.astype(np.uint8)) if isinstance(img, np.ndarray) else img  # from np  
            curr_frames[i] = img_P          
            annotator = Annotator(img_P, line_width=2, pil=not ascii)
            # if self.cfg.STRONGSORT.ECC:  # camera motion compensation
            #     strongsort_list[i].tracker.camera_update(prev_frames[i], curr_frames[i])    
            if hasattr(tracker_list[i], 'tracker') and hasattr(tracker_list[i].tracker, 'camera_update'):
                if prev_frames[i] is not None and curr_frames[i] is not None:  # camera motion compensation
                    tracker_list[i].tracker.camera_update(prev_frames[i], curr_frames[i])
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], img_P.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string

                xywhs = xyxy2xywh(det[:, 0:4])
                confs = det[:, 4]
                clss = det[:, 5]

                # pass detections to strongsort
                t4 = time_sync()
                outputs[i] = tracker_list[i].update(det.cpu(), img_P)
                t5 = time_sync()
                dt[3] += t5 - t4    
                
                if len(outputs[i]) > 0:
                    for j, (output, conf) in enumerate(zip(outputs[i], confs)):
    
                        bboxes = output[0:4]
                        id = output[4]
                        cls = output[5]
                        if int(cls) is 0:
                            testt.append([output[0], output[1], output[2], output[3], output[4]])

                LOGGER.info(f'{s}Done. YOLO:({t3 - t2:.3f}s), StrongSORT:({t5 - t4:.3f}s)')

            else:
                # strongsort_list[i].increment_ages()
                LOGGER.info('No detections')
            # if testt:
            #     pubRos(testt)
            # else:
            #     print("null")      

            img_P = annotator.result()
            cv2.imshow("strongSort", img_P)
            cv2.waitKey(1)  # 1 millisecond
            # if show_vid:
            #     cv2.imshow(str(p), im0)
            #     cv2.waitKey(1)  # 1 millisecond

            prev_frames[i] = curr_frames[i]

        t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
        LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS, %.1fms strong sort update per image at shape {(1, 3, *self.imgsz)}' % t)
        if self.update:
            strip_optimizer(self.yolo_weights)  # update model (to fix SourceChangeWarning)


'''
def pubRos(_test):
    global cnt
    global f_track
    global d_cnt
    global track_miss
    global track_true
    pub = rospy.Publisher('tracker', track, queue_size=1)
    rate = rospy.Rate(10)
    pub_array = track()
    size =0
    x=0
    y=0
    track_x = 0
    track_y = 0
    t_num=0
    not_track = False

    if (cnt - d_cnt) > 30 and track_miss:
        cnt = 0
        d_cnt = 0
        track_miss = False
        track_true = False
    for tmp in _test:
        w = tmp[2]-tmp[0]
        h = tmp[3]-tmp[1]
        if size < w*h:
            size = w*h
            x = (tmp[2]+tmp[0])/2    
            y = (tmp[3]+tmp[1])/2    
            t_num = tmp[4]
        if not cnt == 0 and tmp[4] == f_track:
            track_x = (tmp[2]+tmp[0])/2 
            track_y = (tmp[3]+tmp[1])/2
            not_track = True
            track_true = True


    if cnt == 0 and track_true == False:
        f_track = t_num
        not_track = True

    else:
        x = track_x
        y = track_y
    print("Track person num")
    print(f_track)
    if not_track:
        pub_array.position.append(x)
        pub_array.position.append(y)
        if not rospy.is_shutdown():
            pub.publish(pub_array)
            rate.sleep()
    else:
        if d_cnt < cnt and track_miss == False:
            d_cnt = cnt
            track_miss = True
    cnt += 1
'''
    
def main(args):
    pc = track_person()
    rospy.init_node('parcel_find_server', anonymous=True)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main(sys.argv)
    

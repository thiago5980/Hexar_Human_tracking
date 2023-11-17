#!/usr/bin/env python3
import numpy as np
import cv2
import os
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import csv
import serial
import threading
import time
ser = serial.Serial('/dev/ttyACM0', 9600)
ser.reset_input_buffer()
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
from yolov5.utils.augmentations import letterbox
from PIL import Image, ImageDraw

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov5 strongsort root directory
WEIGHTS = ROOT / 'weights'

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if str(ROOT / 'yolov5') not in sys.path:
    sys.path.append(str(ROOT / 'yolov5'))  # add yolov5 ROOT to PATH
if str(ROOT / 'trackers' / 'strong_sort') not in sys.path:
    sys.path.append(str(ROOT / 'trackers' / 'strong_sort'))  # add strong_sort ROOT to PATH

ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
import pyrealsense2 as rs
import logging
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.dataloaders import VID_FORMATS, LoadImages, LoadStreams
from yolov5.utils.general import (LOGGER, check_img_size, non_max_suppression, scale_coords, check_requirements, cv2,
                                  check_imshow, xyxy2xywh, increment_path, strip_optimizer, colorstr, print_args, check_file)
from yolov5.utils.torch_utils import select_device, time_sync
from yolov5.utils.plots import Annotator, colors, save_one_box
from trackers.multi_tracker_zoo import create_tracker
# remove duplicated stream handler to avoid duplicated logging

def readthread(ser):
    print('thread')
    while True:
        if ser.readable():
            line = ser.readline().decode('utf-8').rstrip()
            print(line)
            time.sleep(0.1)

    ser.close()

logging.getLogger().removeHandler(logging.getLogger().handlers[0])
# Create a pipeline
pipeline = rs.pipeline()

# Create a config and configure the pipeline to stream
#  different resolutions of color and depth streams
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
_device = pipeline_profile.get_device()
device_product_line = str(_device.get_info(rs.camera_info.product_line))

found_rgb = False
for s in _device.sensors:
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

cnt = 0
f_track = 0
d_cnt = 0
track_miss = False
track_true = False
depth_list = []
A = 1
H = 1
Q = 0
R = 4
x_0 = 30
x_esti, P = None, None
def kalman_filter(z_meas, x_esti, P):
    """Kalman Filter Algorithm for One Variable."""
    # (1) Prediction.
    x_pred = A * x_esti
    P_pred = A * P * A + Q
    # (2) Kalman Gain.
    K = P_pred * H / (H * P_pred * H + R)
    # (3) Estimation.
    x_esti = x_pred + K * (z_meas - H * x_pred)
    # (4) Error Covariance.
    P = P_pred - K * H * P_pred
    return x_esti, P

def depth_detect(_test, _fd, _d):
    global cnt
    global f_track
    global d_cnt
    global track_miss
    global track_true   
    global x_0
    global x_esti
    global P
    global depth_list
    if len(depth_list) == 10:
        depth_list.pop(0)
    depth_list.append(_d)
 
    size =0
    x=0
    y=0
    track_x = 0
    track_y = 0
    t_num=0
    not_track = False
#    pub = rospy.Publisher('tracker', Twist, queue_size=1)
#    rate = rospy.Rate(100)
    if (cnt - d_cnt) > 10 and track_miss:
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

    depth_ = _d
    if not_track:
        if len(depth_list) == 10:
            for i, i_depth in enumerate(depth_list):
                detph_ = sum(depth_list)/10.
                if i == 0:
                    x_esti, P = detph_, 2.0
                else:
                    x_esti, P = kalman_filter(i_depth, x_esti, P)
            depth_ = x_esti 
        else: 
            depth_ = _d

#        print("-------in kalman-------")
#        print(x_esti)
#        print("--------------")

#        position.linear.x = _d        # modify depth
#        position.linear.z = _fd       # original
#        position.linear.y = x_esti 
       
#        if not rospy.is_shutdown():
#            pub.publish(position)
#            rate.sleep()

    else:
        #print("track miss")
        if d_cnt < cnt and track_miss == False:
            d_cnt = cnt
            track_miss = True
    cnt += 1
#    print("track_num : " + str(f_track) + "   delay_time : " + str(cnt) +  "  distance : " + str(_d))
    print("Track person num"+str(f_track), "  point_x : ", track_x, "  point_y : ", track_y, "  distance : " + str(depth_))
    return x,y, depth_
as_stop = 0
_angle = 0.
_speed = 0

def pubSpeed(x, cm, stop):
    global as_stop, _angle, _speed
    speed =0
    angle =0. 
    gain = 0.
    # print(x, cm)
    print("stop : ", stop, " as_stop : ", as_stop)
    if stop: 
        as_stop = as_stop + 1
        if as_stop >= 10:
            speed = 0
            angle = 0.
            _speed = 0
            _angle = 0.
            #print("as_stop 0")
        #print("in stop")
    else: 
        as_stop = 0
        #print("out stop")

    if x == 0:
        _speed = _speed
        _angle = _angle
        #print("x=0")
    else:
        temp = abs(x-240) + 0.01
        gain = 0.208 * temp            
        if x <=240: gain = gain * (-1)
            
        if cm <= 30.: _speed = 0
        elif cm > 30. and cm <=300.: _speed = int((cm-30.+0.01)*0.37 + 20)
        else: _speed = 0

        print("in else")
    if abs(gain) > 0:    
        _angle = gain
        print("in gain")
    angle = _angle
    speed = _speed
    send_speed = speed
    send_angle = abs(int(angle))
    print('-------------------',send_angle)   
    #send_speed.data = speed
    #send_angle.data = -angle - 27.
    if angle >= 0.:
        send_msg = bytes('s' + str(send_speed) + 'b' + str(send_angle) + '\n', 'utf-8')
    else:
        send_msg = bytes('s' + str(send_speed) + 'a' + str(send_angle) + '\n', 'utf-8')
    ser.write(send_msg)
    print("sena")
    time.sleep(0.1)                       

            
# Streaming loop
# f = open('last.csv', 'w', newline='')
# wr=csv.writer(f)
send_x = 0
send_y = 0
stop_sign = True
send_depth = 0
try:
    device = select_device('0')
    model = DetectMultiBackend(WEIGHTS / 'yolov5n.pt', device=device, dnn=False, data=None, fp16=False)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size((480, 640), s=stride)  # check image size
    tracker_list = []
    nr_sources = 1
    for i in range(nr_sources):
        tracker = create_tracker('strongsort', WEIGHTS / 'osnet_x0_25_msmt17.pt', device, False)
        tracker_list.append(tracker, )
        if hasattr(tracker_list[i], 'model'):
            if hasattr(tracker_list[i].model, 'warmup'):
                tracker_list[i].model.warmup()
    outputs = [None] * nr_sources   
    dt, seen = [0.0, 0.0, 0.0, 0.0], 0
    curr_frames, prev_frames = [None] * nr_sources, [None] * nr_sources

    thread = threading.Thread(target=readthread, args=(ser,))
    thread.start()    

    while True:
        # Get frameset of color and depth
        # for i in range(60):
        frames = pipeline.wait_for_frames()
        # frames.get_depth_frame() is a 640x360 depth image

        # Align the depth frame to color frame
        aligned_frames = align.process(frames)

        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
        color_frame = aligned_frames.get_color_frame()

        # Validate that both frames are valid
        if not aligned_depth_frame or not color_frame:
            continue

        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        color_image = cv2.rotate(color_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        depth_image = cv2.rotate(depth_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        im = letterbox(color_image, imgsz, stride=stride, auto=pt)[0]
        im = im.transpose((2, 0, 1))[::-1]
        im = np.ascontiguousarray(im)
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if False else im.float()  # uint8 to fp16/32
        im /= 255.0  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        # print(im)
        # print(type(im))
        dt[0] += t2 - t1
        # print(im.dtype) #float32
        pred = model(im, augment=False, visualize=False)
        t3 = time_sync()
        dt[1] += t3 - t2

        pred = non_max_suppression(pred, 0.5, 0.45, 0, False, max_det=1000)
        dt[2] += time_sync() - t3
        # print(color_image.dtype) # uint8
        for i, det in enumerate(pred):  # detections per image
            seen += 1
            testt = []
            # img_P = Image.fromarray(color_image.astype(np.uint8)) if isinstance(color_image, np.ndarray) else color_image  # from np
            img_P = color_image
            curr_frames[i] = img_P
            # print(img_P)
            annotator = Annotator(img_P, line_width=2, pil=not ascii)
            
            if hasattr(tracker_list[i], 'tracker') and hasattr(tracker_list[i].tracker, 'camera_update'):
                if prev_frames[i] is not None and curr_frames[i] is not None:  # camera motion compensation
                    tracker_list[i].tracker.camera_update(prev_frames[i], curr_frames[i])
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], img_P.shape).round()  # xyxy

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    f"{n} {names[int(c)]}{'s' * (n > 1)}, "
                t4 = time_sync()
                outputs[i] = tracker_list[i].update(det.detach().cpu(), img_P)
                t5 = time_sync()
                dt[3] += t5 - t4                

                # draw boxes for visualization
                if len(outputs[i]) > 0:
                    for j, (output, conf) in enumerate(zip(outputs[i], det[:, 4])):
                        bboxes = output[0:4]
                        testt.append([output[0], output[1], output[2], output[3], output[4]])
                        id = output[4]
                        cls = output[5]
                        c = int(cls)  # integer class
                        id = int(id)  # integer id
                        # if c==0:
                        _dx = int((output[0]+output[2])/2)
                        _dy = int((output[1]+output[3])/2)
                        delta_x = int(output[2]-output[0])
                        delta_y = int(output[3]-output[1])
                        d_temp = depth_image[int(output[1]+(delta_x/10)):int(output[3]-(delta_x/10)),int(output[0]+(delta_y/10)):int(output[2]-(delta_y/10))]
                        array_1d = np.array(d_temp).flatten().tolist()
                        # wr.writerow(array_1d)
                        s_ol_data = pd.Series(array_1d)
                        level_1q = s_ol_data.quantile(0.25)
                        level_3q = s_ol_data.quantile(0.75)
                        IQR = level_3q - level_1q
                        rev_range = 3  # 제거 범위 조절 변수
                        dff = s_ol_data[(s_ol_data <= level_3q + (rev_range * IQR)) & (s_ol_data >= level_1q - (rev_range * IQR))]
                        dff = dff.reset_index(drop=True)
                        dff = [i for i in dff if i not in {0.}]
                        depth_mean = np.mean(dff) / 10.
                        print("dx" + str(_dx) + " dy " + str(_dy))
                        if _dx < 480 and _dy < 640:
                            _depth = depth_image.item((_dy, _dx))/10.0
                         #print(color_image.shape)
                            print("person id : "+str(id)+"  person depth : "+str(_depth))
                        label = None if False else (f'{id} {names[c]}' if False else \
                            (f'{id} {conf:.2f}' if False else f'{id} {names[c]} {conf:.2f}'))
                        annotator.box_label(bboxes, label, color=colors(c, True))
                
                LOGGER.info(f'{s}Done. yolo:({t3 - t2:.3f}s), {"strongsort"}:({t5 - t4:.3f}s)')                
            else:
                #strongsort_list[i].increment_ages()
                LOGGER.info('No detections')
                send_stop = True
                send_x = 0
                send_y = 0
                send_depth = 0.
            if testt:
                #print("find")
                send_x, send_y, send_depth = depth_detect(testt, _depth,depth_mean)
                send_stop = False
            else:
                send_stop = True
                send_x = 0
                send_y = 0
                send_depth = 0.
                print("null")
            
            img_P = annotator.result()
            prev_frames[i] = curr_frames[i]
            # temp = depth_image.item((5,5))
            # print(temp)
            cv2.imshow("final", img_P)
            cv2.imshow("depth_img", depth_image)
        # cv2.imshow("color_img", color_image)
        pubSpeed(send_x, send_depth, send_stop)
        key = cv2.waitKey(1)
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            f.close()
            cv2.destroyAllWindows()
            break
finally:
    pipeline.stop()



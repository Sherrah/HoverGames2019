
# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
from functools import partial
import re
import time

import numpy as np
from PIL import Image
import svgwrite
import gstreamer
import sigfox_stub

from pose_engine import PoseEngine

EDGES = (
    ('nose', 'left eye'),
    ('nose', 'right eye'),
    ('nose', 'left ear'),
    ('nose', 'right ear'),
    ('left ear', 'left eye'),
    ('right ear', 'right eye'),
    ('left eye', 'right eye'),
    ('left shoulder', 'right shoulder'),
    ('left shoulder', 'left elbow'),
    ('left shoulder', 'left hip'),
    ('right shoulder', 'right elbow'),
    ('right shoulder', 'right hip'),
    ('left elbow', 'left wrist'),
    ('right elbow', 'right wrist'),
    ('left hip', 'right hip'),
    ('left hip', 'left knee'),
    ('right hip', 'right knee'),
    ('left knee', 'left ankle'),
    ('right knee', 'right ankle'),
)


def shadow_text(dwg, x, y, text, font_size=16):
    dwg.add(dwg.text(text, insert=(x + 1, y + 1), fill='black',
                     font_size=font_size, style='font-family:sans-serif'))
    dwg.add(dwg.text(text, insert=(x, y), fill='white',
                     font_size=font_size, style='font-family:sans-serif'))


def draw_pose(dwg, pose, color='yellow', threshold=0.2):
    xys = {}
    for label, keypoint in pose.keypoints.items():
        if keypoint.score < threshold: continue
        xys[label] = (int(keypoint.yx[1]), int(keypoint.yx[0]))
        dwg.add(dwg.circle(center=(int(keypoint.yx[1]), int(keypoint.yx[0])), r=5,
                           fill='cyan', fill_opacity=keypoint.score, stroke=color))

    for a, b in EDGES:
        if a not in xys or b not in xys: continue
        ax, ay = xys[a]
        bx, by = xys[b]
        dwg.add(dwg.line(start=(ax, ay), end=(bx, by), stroke=color, stroke_width=2))


def run(callback, use_appsrc=False):
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--mirror', help='flip video horizontally', action='store_true')
    parser.add_argument('--model', help='.tflite model path.', required=False)
    parser.add_argument('--res', help='Resolution', default='640x480',
                        choices=['480x360', '640x480', '1280x720'])
    parser.add_argument('--videosrc', help='Which video source to use', default='/dev/video1')
    parser.add_argument('--h264', help='Use video/x-h264 input', action='store_true')
    args = parser.parse_args()

    default_model = 'models/posenet_mobilenet_v1_075_%d_%d_quant_decoder_edgetpu.tflite'
    if args.res == '480x360':
        src_size = (640, 480)
        appsink_size = (480, 360)
        model = args.model or default_model % (353, 481)
    elif args.res == '640x480':
        src_size = (640, 480)
        appsink_size = (640, 480)
        model = args.model or default_model % (481, 641)
    elif args.res == '1280x720':
        src_size = (1280, 720)
        appsink_size = (1280, 720)
        model = args.model or default_model % (721, 1281)

    print('Loading model: ', model)
    engine = PoseEngine(model, mirror=args.mirror)
    gstreamer.run_pipeline(partial(callback, engine),
                           src_size, appsink_size,
                           use_appsrc=use_appsrc, mirror=args.mirror,
                           videosrc=args.videosrc, h264input=args.h264)


def main():
    last_time = time.monotonic()
    n = 0
    sum_fps = 0
    sum_process_time = 0
    sum_inference_time = 0
    debug_counter = 0
    last_timestamp = 0
    
    def render_overlay(engine, image, svg_canvas):
        r_eye_x = 0
        r_eye_y = 0
        r_eye_score = 0
        l_eye_x = 0
        l_eye_y = 0
        l_eye_score = 0
        l_wrist_x = 0
        l_wrist_y = 0
        l_wrist_score = 0
        count = 0

        nonlocal n, sum_fps, sum_process_time, sum_inference_time, last_time, debug_counter, last_timestamp
        start_time = time.monotonic()
        outputs, inference_time = engine.DetectPosesInImage(image)
        end_time = time.monotonic()
        n += 1
        sum_fps += 1.0 / (end_time - last_time)
        sum_process_time += 1000 * (end_time - start_time) - inference_time
        sum_inference_time += inference_time
        last_time = end_time
        text_line = 'PoseNet: %.1fms Frame IO: %.2fms TrueFPS: %.2f Nposes %d' % (
            sum_inference_time / n, sum_process_time / n, sum_fps / n, len(outputs)
        )
        #print(text_line)

        shadow_text(svg_canvas, 10, 20, text_line)
        for pose in outputs:
            draw_pose(svg_canvas, pose)
            #print('\nPose Score: ', pose.score)
            
#            for label, keypoint in pose.keypoints.items():
 #               print("%s\t %-3d %-3d %1.3f" % (label, keypoint.yx[1], keypoint.yx[0], keypoint.score))

            for label, keypoint in pose.keypoints.items():
                if label == 'right eye':
                    r_eye_x = keypoint.yx[1]
                    r_eye_y = keypoint.yx[0]
                    r_eye_score = keypoint.score
                if label == 'left eye':
                    l_eye_x = keypoint.yx[1]
                    l_eye_y = keypoint.yx[0]
                    l_eye_score = keypoint.score
                if label == 'left wrist':
                    l_wrist_x = keypoint.yx[1]
                    l_wrist_y = keypoint.yx[0]
                    l_wrist_score = keypoint.score
                    if l_wrist_score > 0.5:
                        if l_wrist_y < l_eye_y:
                            debug_counter += 1
                            current_timestamp = time.time()
                            if current_timestamp - last_timestamp > 1:
                                last_timestamp = current_timestamp
                                print('!!!!!!!!!!                            H     E     L     P                      !!!!!!!!!!!')
                                count += 1
                                sigfox_stub.sig_transmit(count)
#                        print('w2e_dis/e2e_dis: %-4d' % ((l_wrist_x-l_eye_x)/(l_eye_x-r_eye_x)) )
#                        print('eye_dis: %-4d\nw2e_dis: %-4d' % (l_eye_x-r_eye_x,l_wrist_x-l_eye_x))
#                        print('eye_dis:%-4d\nl_e:[%-4d, %-4d]\nl_w [%-4d, %-4d]' % (l_eye_x-r_eye_x,l_eye_x,l_eye_y,l_wrist_x, l_wrist_y))
#                        print('l_e:[%-4d, %-4d]\nl_w [%-4d, %-4d]' % (l_eye_x,l_eye_y,l_wrist_x, l_wrist_y))
        #current_timestamp = time.time()
        #print("Current timestamp: %3.4f" % current_timestamp)
        time.sleep(0.1)
        print("Debug Counter: %d" % debug_counter)
    run(render_overlay)

if __name__ == '__main__':
    main()

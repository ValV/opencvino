#!/usr/bin/env python3
"""
 Copyright (C) 2018-2022 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import logging as log
import sys

from configparser import ConfigParser
from os import path as osp, access, R_OK
from time import perf_counter

import cv2 as cv
# import numpy as np
# from threading import Thread
# if sys.version_info[0] == 2:
#     import Queue as queue
# else:
#     import queue
#
# from common import *
#
# sys.path.append(str(Path(__file__).resolve().parents[2] / 'common/python'))
# sys.path.append(str(Path(__file__).resolve().parents[2] / 'common/python/openvino/model_zoo'))

from model_api.models import DetectionModel, DetectionWithLandmarks, RESIZE_TYPES, OutputTransform
from model_api.performance_metrics import PerformanceMetrics
from model_api.pipelines import get_user_config, AsyncPipeline
from model_api.adapters import create_core, OpenvinoAdapter, OVMSAdapter

# import monitors

from images_capture import open_images_capture
from helpers import resolution, log_latency_per_stage
from visualizers import ColorPalette

from cvdnn import inference_opencv, backends, targets

log.basicConfig(format='[ %(levelname)s ] %(message)s', level=log.DEBUG, stream=sys.stdout)


def build_argparser():
    from argparse import ArgumentParser, SUPPRESS

    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')
    # args.add_argument('-m', '--model', required=True,
    #                   help='Required. Path to an .xml file with a trained model '
    #                        'or address of model inference service if using ovms adapter.')
    # available_model_wrappers = [name.lower() for name in DetectionModel.available_wrappers()]
    # args.add_argument('-at', '--architecture_type', help='Required. Specify model\' architecture type.',
    #                   type=str, required=True, choices=available_model_wrappers)
    # args.add_argument('--adapter', help='Optional. Specify the model adapter. Default is openvino.',
    #                   default='openvino', type=str, choices=('openvino', 'ovms'))
    args.add_argument('-i', '--input', required=True,
                      help='Required. An input to process. The input must be a single image, '
                           'a folder of images, video file or camera id.')
    args.add_argument('-d', '--device', default='CPU', type=str,
                      help='Optional. Specify the target device to infer on; CPU, GPU, HDDL or MYRIAD is '
                           'acceptable. The demo will look for a suitable plugin for device specified. '
                           'Default value is CPU.')

    common_model_args = parser.add_argument_group('Common model options')
    common_model_args.add_argument('--labels', help='Optional. Labels mapping file.', default=None, type=str)
    common_model_args.add_argument('-t', '--prob_threshold', default=0.5, type=float,
                                   help='Optional. Probability threshold for detections filtering.')
    common_model_args.add_argument('--resize_type', default=None, choices=RESIZE_TYPES.keys(),
                                   help='Optional. A resize type for model preprocess. '
                                        'By default used model predefined type.')
    common_model_args.add_argument('--input_size', default=(600, 600), type=int, nargs=2,
                                   help='Optional. The first image size used for CTPN model reshaping. '
                                        'Default: 600 600. Note that submitted images should have the same resolution, '
                                        'otherwise predictions might be incorrect.')
    common_model_args.add_argument('--anchors', default=None, type=float, nargs='+',
                                   help='Optional. A space separated list of anchors. '
                                        'By default used default anchors for model. Only for YOLOV4 architecture type.')
    common_model_args.add_argument('--masks', default=None, type=int, nargs='+',
                                   help='Optional. A space separated list of mask for anchors. '
                                        'By default used default masks for model. Only for YOLOV4 architecture type.')
    common_model_args.add_argument('--layout', type=str, default=None,
                                   help='Optional. Model inputs layouts. '
                                        'Ex. NCHW or input0:NCHW,input1:NC in case of more than one input.')
    common_model_args.add_argument('--num_classes', default=None, type=int,
                                   help='Optional. Number of detected classes. Only for NanoDet, NanoDetPlus '
                                        'architecture types.')

    infer_args = parser.add_argument_group('Inference options')
    infer_args.add_argument('-nireq', '--num_infer_requests', help='Optional. Number of infer requests',
                            default=0, type=int)
    infer_args.add_argument('-nstreams', '--num_streams',
                            help='Optional. Number of streams to use for inference on the CPU or/and GPU in throughput '
                                 'mode (for HETERO and MULTI device cases use format '
                                 '<device1>:<nstreams1>,<device2>:<nstreams2> or just <nstreams>).',
                            default='', type=str)
    infer_args.add_argument('-nthreads', '--num_threads', default=None, type=int,
                            help='Optional. Number of threads to use for inference on CPU (including HETERO cases).')

    io_args = parser.add_argument_group('Input/output options')
    io_args.add_argument('--loop', default=False, action='store_true',
                         help='Optional. Enable reading the input in a loop.')
    io_args.add_argument('-o', '--output', required=False,
                         help='Optional. Name of the output file(s) to save.')
    io_args.add_argument('-limit', '--output_limit', required=False, default=1000, type=int,
                         help='Optional. Number of frames to store in output. '
                              'If 0 is set, all frames are stored.')
    io_args.add_argument('--no_show', help="Optional. Don't show output.", action='store_true')
    io_args.add_argument('--output_resolution', default=None, type=resolution,
                         help='Optional. Specify the maximum output window resolution '
                              'in (width x height) format. Example: 1280x720. '
                              'Input frame size used by default.')
    io_args.add_argument('-u', '--utilization_monitors', default='', type=str,
                         help='Optional. List of monitors to show initially.')

    input_transform_args = parser.add_argument_group('Input transform options')
    input_transform_args.add_argument('--reverse_input_channels', default=False, action='store_true',
                                      help='Optional. Switch the input channels order from '
                                           'BGR to RGB.')
    input_transform_args.add_argument('--mean_values', default=None, type=float, nargs=3,
                                      help='Optional. Normalize input by subtracting the mean '
                                           'values per channel. Example: 255.0 255.0 255.0')
    input_transform_args.add_argument('--scale_values', default=None, type=float, nargs=3,
                                      help='Optional. Divide input by scale values per channel. '
                                           'Division is applied after mean values subtraction. '
                                           'Example: 255.0 255.0 255.0')

    debug_args = parser.add_argument_group('Debug options')
    debug_args.add_argument('-r', '--raw_output_message', help='Optional. Output inference results raw values showing.',
                            default=False, action='store_true')

    opencv_args = parser.add_argument_group('OpenCV DNN options')
    opencv_args.add_argument('--zoo', default=osp.join(osp.dirname(osp.abspath(__file__)), 'models.yml'),
                             help='An optional path to file with preprocessing parameters.')
    opencv_args.add_argument('--framework', choices=['caffe', 'torch', 'darknet', 'dldt'],
                             help='Optional name of an origin framework of the model. '
                                  'Detect it automatically if it does not set.')
    # opencv_args.add_argument('--thr', type=float, default=0.5, help='Confidence threshold')
    opencv_args.add_argument('--nms', type=float, default=0.4, help='Non-maximum suppression threshold')
    opencv_args.add_argument('--backend', choices=backends, default=cv.dnn.DNN_BACKEND_DEFAULT, type=int,
                             help="Choose one of computation backends: "
                                  "%d: automatically (by default), "
                                  "%d: Halide language (http://halide-lang.org/), "
                                  "%d: Intel's Deep Learning Inference Engine "
                                  "(https://software.intel.com/openvino-toolkit), "
                                  "%d: OpenCV implementation, "
                                  "%d: VKCOM, "
                                  "%d: CUDA" % backends)
    opencv_args.add_argument('--target', choices=targets, default=cv.dnn.DNN_TARGET_CPU, type=int,
                             help='Choose one of target computation devices: '
                                  '%d: CPU target (by default), '
                                  '%d: OpenCL, '
                                  '%d: OpenCL fp16 (half-float precision), '
                                  '%d: NCS2 VPU, '
                                  '%d: HDDL VPU, '
                                  '%d: Vulkan, '
                                  '%d: CUDA, '
                                  '%d: CUDA fp16 (half-float preprocess)' % targets)
    opencv_args.add_argument('--async', type=int, default=0,
                             dest='asyncN',
                             help='Number of asynchronous forwards at the same time. '
                                  'Choose 0 for synchronous mode')
    opencv_args.add_argument('--classes', required=False, default=None,
                             help='Optional path to a text file with names of classes to label detected objects.')

    return parser


def draw_detections(frame, detections, palette, labels, output_transform):
    frame = output_transform.resize(frame)
    for detection in detections:
        class_id = int(detection.id)
        color = palette[class_id]
        det_label = labels[class_id] if labels and len(labels) >= class_id else '#{}'.format(class_id)
        xmin, ymin, xmax, ymax = detection.get_coords()
        xmin, ymin, xmax, ymax = output_transform.scale([xmin, ymin, xmax, ymax])
        cv.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
        cv.putText(frame, '{} {:.1%}'.format(det_label, detection.score),
                   (xmin, ymin - 7), cv.FONT_HERSHEY_COMPLEX, 0.6, color, 1)
        if isinstance(detection, DetectionWithLandmarks):
            for landmark in detection.landmarks:
                landmark = output_transform.scale(landmark)
                cv.circle(frame, (int(landmark[0]), int(landmark[1])), 2, (0, 255, 255), 2)
    return frame


def print_raw_results(detections, labels, frame_id):
    log.debug(' ------------------- Frame # {} ------------------ '.format(frame_id))
    log.debug(' Class ID | Confidence | XMIN | YMIN | XMAX | YMAX ')
    for detection in detections:
        xmin, ymin, xmax, ymax = detection.get_coords()
        class_id = int(detection.id)
        det_label = labels[class_id] if labels and len(labels) >= class_id else '#{}'.format(class_id)
        log.debug('{:^9} | {:10f} | {:4} | {:4} | {:4} | {:4} '
                  .format(det_label, detection.score, xmin, ymin, xmax, ymax))


def inference_openvino(args, cap):
    if args.architecture_type != 'yolov4' and args.anchors:
        log.warning(
            'The "--anchors" option works only for "-at==yolov4". '
            'Option will be omitted'
        )
    if args.architecture_type != 'yolov4' and args.masks:
        log.warning(
            'The "--masks" option works only for "-at==yolov4". '
            'Option will be omitted'
        )
    if args.architecture_type not in ['nanodet', 'nanodet-plus'] and args.num_classes:
        log.warning(
            'The "--num_classes" option works only for "-at==nanodet" and '
            '"-at==nanodet-plus". Option will be omitted'
        )

    if args.adapter == 'openvino':
        plugin_config = get_user_config(
            args.device, args.num_streams, args.num_threads
        )
        model_adapter = OpenvinoAdapter(
            create_core(), args.model, device=args.device,
            plugin_config=plugin_config,
            max_num_requests=args.num_infer_requests,
            model_parameters={'input_layouts': args.layout}
        )
    elif args.adapter == 'ovms':
        model_adapter = OVMSAdapter(args.model)
    else:
        model_adapter = None

    configuration = {
        'resize_type': args.resize_type,
        'mean_values': args.mean_values,
        'scale_values': args.scale_values,
        'reverse_input_channels': args.reverse_input_channels,
        'path_to_labels': args.labels,
        'confidence_threshold': args.prob_threshold,
        'input_size': args.input_size,  # The CTPN specific
        'num_classes': args.num_classes,  # The NanoDet and NanoDetPlus specific
    }
    model = DetectionModel.create_model(
        args.architecture_type, model_adapter, configuration
    )
    model.log_layers_info()

    detector_pipeline = AsyncPipeline(model)

    next_frame_id = 0
    next_frame_id_to_show = 0

    palette = ColorPalette(len(model.labels) if model.labels else 100)
    metrics = PerformanceMetrics()
    render_metrics = PerformanceMetrics()
    # presenter = None
    output_transform = None
    video_writer = cv.VideoWriter()

    # OpenVINO inference loop start
    while True:
        if detector_pipeline.callback_exceptions:
            raise detector_pipeline.callback_exceptions[0]
        # Process all completed requests
        results = detector_pipeline.get_result(next_frame_id_to_show)
        if results:
            objects, frame_meta = results
            frame = frame_meta['frame']
            start_time = frame_meta['start_time']

            if len(objects) and args.raw_output_message:
                print_raw_results(objects, model.labels, next_frame_id_to_show)

            # presenter.drawGraphs(frame)
            rendering_start_time = perf_counter()
            frame = draw_detections(
                frame, objects, palette, model.labels, output_transform
            )
            render_metrics.update(rendering_start_time)
            metrics.update(start_time, frame)

            if (video_writer.isOpened() and (
                    args.output_limit <= 0 or
                    next_frame_id_to_show <= args.output_limit - 1
            )):
                video_writer.write(frame)
            next_frame_id_to_show += 1

            if not args.no_show:
                cv.imshow('Detection Results', frame)
                key = cv.waitKey(1)

                ESC_KEY = 27
                # Quit.
                if key in {ord('q'), ord('Q'), ESC_KEY}:
                    break
                # presenter.handleKey(key)
            continue

        if detector_pipeline.is_ready():
            # Get new image/frame
            start_time = perf_counter()
            frame = cap.read()
            if frame is None:
                if next_frame_id == 0:
                    raise ValueError("Can't read an image from the input")
                break
            if next_frame_id == 0:
                output_transform = OutputTransform(frame.shape[:2],
                                                   args.output_resolution)
                if args.output_resolution:
                    output_resolution = output_transform.new_resolution
                else:
                    output_resolution = (frame.shape[1], frame.shape[0])
                # presenter = monitors.Presenter(args.utilization_monitors, 55,
                #                                (round(output_resolution[0] / 4),
                #                                 round(output_resolution[1] / 8)))
                if args.output and not video_writer.open(
                        args.output, cv.VideoWriter_fourcc(*'MJPG'),
                        cap.fps(), output_resolution
                ):
                    raise RuntimeError("Can't open video writer")
            # Submit for inference
            detector_pipeline.submit_data(
                frame, next_frame_id, {'frame': frame, 'start_time': start_time}
            )
            next_frame_id += 1
        else:
            # Wait for empty request
            detector_pipeline.await_any()
    # OpenVINO inference loop stop

    # OpenVINO postprocessing start
    detector_pipeline.await_all()
    if detector_pipeline.callback_exceptions:
        raise detector_pipeline.callback_exceptions[0]
    # Process completed requests
    for next_frame_id_to_show in range(next_frame_id_to_show, next_frame_id):
        results = detector_pipeline.get_result(next_frame_id_to_show)
        objects, frame_meta = results
        frame = frame_meta['frame']
        start_time = frame_meta['start_time']

        if len(objects) and args.raw_output_message:
            print_raw_results(objects, model.labels, next_frame_id_to_show)

        # presenter.drawGraphs(frame)
        rendering_start_time = perf_counter()
        frame = draw_detections(
            frame, objects, palette, model.labels, output_transform
        )
        render_metrics.update(rendering_start_time)
        metrics.update(start_time, frame)

        if (video_writer.isOpened() and (
                args.output_limit <= 0 or
                next_frame_id_to_show <= args.output_limit - 1
        )):
            video_writer.write(frame)

        if not args.no_show:
            cv.imshow('Detection Results', frame)
            key = cv.waitKey(1)

            ESC_KEY = 27
            # Quit
            if key in {ord('q'), ord('Q'), ESC_KEY}:
                break
            # presenter.handleKey(key)
    # OpenVINO postprocessing stop

    # OpenVINO results begin
    metrics.log_total()
    log_latency_per_stage(cap.reader_metrics.get_latency(),
                          detector_pipeline.preprocess_metrics.get_latency(),
                          detector_pipeline.inference_metrics.get_latency(),
                          detector_pipeline.postprocess_metrics.get_latency(),
                          render_metrics.get_latency())
    # for rep in presenter.reportMeans():
    #     log.info(rep)
    # OpenVINO results end


# def inference_opencv(args, cap):
#     pipeline = WrapperDNN(args, cap)
#     if not args.no_show:
#         pipeline.init_gui()
#     pipeline.inference()


def main():
    OPENVINO_MODEL = 'person-detection-0200'
    OPENVINO_PATH = f'models/intel/{OPENVINO_MODEL}/FP32'
    OPENVINO_NAMES = [f'{OPENVINO_MODEL}.bin', f'{OPENVINO_MODEL}.xml']
    model_openvino = tuple(map(lambda name: osp.join(OPENVINO_PATH, name),
                               OPENVINO_NAMES))
    OPENCV_MODEL = 'yolov4-tiny'
    OPENCV_PATH = 'models/alexeyab'
    OPENCV_NAMES = [f'{OPENCV_MODEL}.weights', f'{OPENCV_MODEL}.cfg']
    model_opencv = tuple(map(lambda name: osp.join(OPENCV_PATH, name),
                             OPENCV_NAMES))

    args = build_argparser().parse_args()
    cap = open_images_capture(args.input, args.loop)

    # OpenVINO
    _, args.model = model_openvino
    output = args.output
    if output and isinstance(output, str):
        split = list(osp.splitext(output))
        split.insert(1, f'.{OPENVINO_MODEL}')
        args.output = ''.join(split)
        log.info(f"Saving to {args.output}...")
    else:
        log.warning(f"Malformed output: {args.output}! Setting to 'None'")
        args.output = None
    args.architecture_type = 'ssd'
    args.adapter = 'openvino'
    inference_openvino(args, cap)

    # cap = cap.cap
    if hasattr(cap, 'cap'):
        cap.cap.set(cv.CAP_PROP_POS_FRAMES, 0)
    elif hasattr(cap, 'file_id'):
        cap.file_id = 0

    # OpenCV
    args.model, args.config = model_opencv
    if output and isinstance(output, str):
        split = list(osp.splitext(output))
        split.insert(1, f'.{OPENCV_MODEL}')
        args.output = ''.join(split)
        log.info(f"Saving to {args.output}...")
    else:
        log.warning(f"Malformed output: {args.output}! Will not write to file!")
        args.output = None
    config = ConfigParser(strict=False)
    config.read(args.config)
    args.backend = cv.dnn.DNN_BACKEND_OPENCV
    args.framework = 'darknet'
    args.nms = 0.15
    args.alias = None
    args.mean = [0, 0, 0]
    args.scale = 1.0
    args.width = int(config['net']['width'])
    args.height = int(config['net']['height'])
    args.rgb = True
    path_coco_names = osp.abspath('coco.names')
    if osp.isfile(path_coco_names) and access(path_coco_names, R_OK):
        print(f"Class names = {path_coco_names}")
        args.classes = path_coco_names
    else:
        args.classes = None
    args.coi = (1, 2, 3)  # class 1 - person, 2 - bike, 3 - car
    # inference_opencv(args, cap)
    inference_opencv(args, cap, PerformanceMetrics, ColorPalette)

    try:
        cap.cap.release()
    except AttributeError:
        try:
            cap.release()
        except:
            pass


if __name__ == '__main__':
    sys.exit(main() or 0)

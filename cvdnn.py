import logging as log

from configparser import ConfigParser
from dataclasses import dataclass
from os import path as osp, access, R_OK
from time import perf_counter
from xml.etree import ElementTree as etree

import cv2 as cv
import numpy as np

from images_capture import open_images_capture


@dataclass
class Args:
    ...


def log_latency_per_stage(*pipeline_metrics):
    stages = ('Decoding', 'Preprocessing', 'Inference', 'Postprocessing', 'Rendering')
    for stage, latency in zip(stages, pipeline_metrics):
        log.info('\t{}:\t{:.1f} ms'.format(stage, latency))


def print_raw_results(frame_id, class_id=0, score=0, left=0, top=0,
                      right=0, bottom=0, classes=None, header=False):
    if header:
        log.debug(' ------------------- Frame # {} ------------------ '.format(frame_id))
        log.debug(' Class ID | Confidence | XMIN | YMIN | XMAX | YMAX ')
    elif all((right, bottom)):
        xmin, ymin, xmax, ymax = left, top, right, bottom
        det_label = (
            classes[class_id]
            if classes and len(classes) >= class_id
            else '#{}'.format(class_id)
        )
        log.debug('{:^9} | {:10f} | {:4} | {:4} | {:4} | {:4} '
                  .format(det_label, score, xmin, ymin, xmax, ymax))


def draw_predictions(frame, classId, conf, left, top, right, bottom,
                     classes=None, palette=None):
    # Draw a bounding box
    # cv.rectangle(frame, (left, top), (right, bottom), (0, 255, 0))

    # label = '%.2f' % conf

    # Print a label of class
    # if isinstance(classes, (tuple, list)):
    #     assert (classId < len(classes))
    #     label = '%s: %s' % (classes[classId], label)

    # labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    # top = max(top, labelSize[1])
    # cv.rectangle(frame, (left, top - labelSize[1]),
    #              (left + labelSize[0], top + baseLine),
    #              (255, 255, 255), cv.FILLED)
    # cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

    try:
        color = palette[classId]
    except IndexError:
        color = (0, 255, 0)

    det_label = classes[classId] if classes and len(classes) >= classId else '#{}'.format(classId)
    cv.rectangle(frame, (left, top), (right, bottom), color, 2)
    cv.putText(frame, '{} {:.1%}'.format(det_label, conf),
               (left, top - 7), cv.FONT_HERSHEY_COMPLEX, 0.6, color, 1)
    return frame


def inference_opencv(args, cap, pmc=None, cpc=None):
    ColorPalette = cpc
    if pmc is not None:
        PerformanceMetrics = pmc

        metrics = PerformanceMetrics()
        metrics_preprocess = PerformanceMetrics()
        metrics_inference = PerformanceMetrics()
        metrics_postprocess = PerformanceMetrics()
        metrics_rendering = PerformanceMetrics()
    else:
        log.warning("Unable to use performance metrics!")

        metrics = None
        metrics_preprocess = None
        metrics_inference = None
        metrics_postprocess = None
        metrics_rendering = None

    if hasattr(cap, 'cap'):
        fps = cap.fps()
        # cap = cap.cap
        targetWidth = int(cap.cap.get(cv.CAP_PROP_FRAME_WIDTH))
        targetHeight = int(cap.cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    elif hasattr(cap, 'get'):
        fps = cap.get(cv.CAP_PROP_FPS)
        targetWidth = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        targetHeight = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    elif hasattr(cap, 'image'):
        fps = 1
        targetHeight, targetWidth = cap.image.shape[:2]
    else:
        fps = -1
        targetWidth = None
        targetHeight = None

    if args.output and targetWidth and targetWidth:
        target = cv.VideoWriter()
        target.open(args.output, cv.VideoWriter_fourcc(*'MJPG'),
                    fps if 0 < fps < 55 else 15.0, (targetWidth, targetHeight))
        # print(f"DEBUG: output shape = {(self.outWidth, self.outHeight)}...")
    else:
        target = None
        targetWidth = None
        targetHeight = None

    if hasattr(args, 'coi') and isinstance(args.coi, (dict, list, set, tuple)):
        args.coi = np.array(tuple(args.coi), dtype=np.uint)
        args.coi -= 1
    else:
        args.coi = None

    classes = None
    if args.classes:
        with open(args.classes, 'rt') as names:
            classes = names.read().rstrip('\n').split('\n')
    if ColorPalette is not None:
        palette = ColorPalette(len(classes) if classes else 2)
    else:
        palette = None

    # Load a network
    net = cv.dnn.readNet(cv.samples.findFile(args.model),
                         cv.samples.findFile(args.config),
                         args.framework)
    net.setPreferableBackend(args.backend)
    net.setPreferableTarget(args.target)
    outNames = net.getUnconnectedOutLayersNames()

    if not args.no_show:
        winName = 'Deep learning object detection in OpenCV'
        # cv.namedWindow(winName, cv.WINDOW_NORMAL)
        cv.namedWindow(winName, cv.WINDOW_AUTOSIZE)
    else:
        winName = None

    # Inference + postprocessing loop (whole video)
    index_frame = 0
    while True:
        # hasFrame, frame = cap.read()
        # if not hasFrame:
        #     break

        time_start = perf_counter()
        frame = cap.read()
        try:
            hasFrame, frame = frame
        except (ValueError, TypeError):
            pass

        # if frame is None:
        #     break

        time_start_preprocessing = perf_counter()
        if frame is not None:
            frameHeight = frame.shape[0]
            frameWidth = frame.shape[1]

            # Create a 4D blob from a frame.
            inpWidth = args.width if args.width else frameWidth
            inpHeight = args.height if args.height else frameHeight
            # print(f"DEBUG: args dims = {args.width, args.height}")
            frame = cv.resize(frame, (inpWidth, inpHeight))
            blob = cv.dnn.blobFromImage(
                frame, size=(inpWidth, inpHeight),
                swapRB=args.rgb, ddepth=cv.CV_8U
            )
            # processedFramesQueue.put(frame)
            if metrics_preprocess is not None:
                metrics_preprocess.update(time_start_preprocessing)

            # Run a model
            time_start_inference = perf_counter()
            net.setInput(blob, scalefactor=args.scale, mean=args.mean)
            if net.getLayer(0).outputNameToIndex('im_info') != -1:  # Faster-RCNN or R-FCN
                net.setInput(
                    np.array([[inpHeight, inpWidth, 1.6]],
                             dtype=np.float32),
                    'im_info'
                )

            outs = net.forward(outNames)
            # predictionsQueue.put(np.copy(outs))
            if metrics_inference is not None:
                metrics_inference.update(time_start_inference)
            if metrics is not None:
                metrics.update(time_start, frame)

            # Postprocessing
            time_start_postprocessing = perf_counter()

            frameHeight = frame.shape[0]
            frameWidth = frame.shape[1]
            # print(f"DEBUG: post-frame dimensions = {frameWidth, frameHeight}")

            layerNames = net.getLayerNames()
            lastLayerId = net.getLayerId(layerNames[-1])
            lastLayer = net.getLayer(lastLayerId)

            confThreshold = args.prob_threshold  # confThreshold
            nmsThreshold = args.nms  # nmsThreshold
            coi = args.coi

            classIds = []
            confidences = []
            boxes = []
            if lastLayer.type == 'DetectionOutput':
                # Network produces output blob with a shape 1x1xNx7 where N is a number of
                # detections and an every detection is a vector of values
                # [batchId, classId, confidence, left, top, right, bottom]
                for out in outs:
                    for detection in out[0, 0]:
                        confidence = detection[2]
                        if confidence > confThreshold:
                            left = int(detection[3])
                            top = int(detection[4])
                            right = int(detection[5])
                            bottom = int(detection[6])
                            width = right - left + 1
                            height = bottom - top + 1
                            if width <= 2 or height <= 2:
                                left = int(detection[3] * frameWidth)
                                top = int(detection[4] * frameHeight)
                                right = int(detection[5] * frameWidth)
                                bottom = int(detection[6] * frameHeight)
                                width = right - left + 1
                                height = bottom - top + 1
                            classIds.append(int(detection[1]) - 1)  # Skip background label
                            confidences.append(float(confidence))
                            boxes.append([left, top, width, height])
            elif lastLayer.type == 'Region':
                # Network produces output blob with a shape NxC where N is a number of
                # detected objects and C is a number of classes + 4 where the first 4
                # numbers are [center_x, center_y, width, height]
                for out in outs:
                    for detection in out:
                        scores = detection[5:]
                        if coi is not None and (scores[coi] < 1e-9).all():
                            continue
                        classId = np.argmax(scores)
                        confidence = scores[classId]
                        # TODO: aggregate with coi
                        if confidence > confThreshold:
                            center_x = int(detection[0] * frameWidth)
                            center_y = int(detection[1] * frameHeight)
                            width = int(detection[2] * frameWidth)
                            height = int(detection[3] * frameHeight)
                            left = int(center_x - width / 2)
                            top = int(center_y - height / 2)
                            classIds.append(classId)
                            confidences.append(float(confidence))
                            boxes.append([left, top, width, height])
            else:
                print('Unknown output layer type: ' + lastLayer.type)
                exit()

            # NMS is used inside Region layer only on DNN_BACKEND_OPENCV for another backends we need NMS in sample
            # or NMS is required if number of outputs > 1
            if len(outNames) > 1 or lastLayer.type == 'Region' and args.backend != cv.dnn.DNN_BACKEND_OPENCV:
                indices = []
                classIds = np.array(classIds)
                # print(f"DEBUG: ids.shape = {classIds.shape}")
                boxes = np.array(boxes)
                # print(f"DEBUG: boxes.shape = {boxes.shape}")
                confidences = np.array(confidences)
                # print(f"DEBUG: confidences.shape = {confidences.shape}")
                unique_classes = set(classIds)
                # print(f"DEBUG: unique classes = {unique_classes}")
                for cl in unique_classes:
                    class_indices = np.where(classIds == cl)[0]
                    # print(f"DEBUG: class indices = {class_indices}")
                    conf = confidences[class_indices]
                    box = boxes[class_indices].tolist()
                    nms_indices = cv.dnn.NMSBoxes(box, conf, confThreshold,
                                                  nmsThreshold)
                    # print(f"DEBUG: nms_indices.shape = {nms_indices.shape}")
                    # print(f"DEBUG: nms_indices = {nms_indices}")
                    # nms_indices = nms_indices[:, 0] if len(nms_indices) else []
                    nms_indices = nms_indices[:, 0] if nms_indices.ndim > 1 else nms_indices
                    indices.extend(class_indices[nms_indices])
            else:
                indices = np.arange(0, len(classIds))
            if metrics_postprocess is not None:
                metrics_postprocess.update(time_start_postprocessing)

            # Rendering
            time_start_rendering = perf_counter()

            if targetWidth and targetHeight:
                # print(f"DEBUG: target (width, height) = {targetWidth, targetHeight}")
                frame = cv.resize(frame, (targetWidth, targetHeight), cv.INTER_NEAREST)

            if args.raw_output_message:
                print_raw_results(index_frame, header=True)
            for i in indices:
                box = boxes[i]
                left = box[0]
                top = box[1]
                width = box[2]
                height = box[3]
                if not all((0 <= left < frameWidth, 0 <= top < frameHeight,
                            0 < left + width <= frameWidth,
                            0 < top + height <= frameHeight)):
                    continue
                if targetWidth and targetHeight:
                    left = round(left / frameWidth * targetWidth)
                    top = round(top / frameHeight * targetHeight)
                    width = round(width / frameWidth * targetWidth)
                    height = round(height / frameHeight * targetHeight)
                # print(f"DEBUG: rescaling (l, t, r, b) = {left, top, left + width, top + height}")
                draw_predictions(frame, classIds[i], confidences[i], left, top,
                                 left + width, top + height, classes=classes,
                                 palette=palette)
                if args.raw_output_message:
                    print_raw_results(index_frame, classIds[i], confidences[i],
                                      left, top, left + width, top + height,
                                      classes)
            if metrics_rendering is not None:
                metrics_rendering.update(time_start_rendering)

            index_frame += 1

            if target is not None and target.isOpened():
                target.write(frame)

            if winName:
                cv.imshow(winName, frame)
                key = cv.waitKey(1)

                ESC_KEY = 27
                # Quit
                if key in {ord('q'), ord('Q'), ESC_KEY}:
                    break
        else:
            break

    if metrics is not None:
        metrics.log_total()
    if all((
            hasattr(cap, 'reader_metrics') and cap.reader_metrics is not None,
            metrics_preprocess is not None,
            metrics_inference is not None,
            metrics_postprocess is not None,
            metrics_rendering is not None,
            index_frame
    )):
        log_latency_per_stage(cap.reader_metrics.get_latency(),
                              metrics_preprocess.get_latency(),
                              metrics_inference.get_latency(),
                              metrics_postprocess.get_latency(),
                              metrics_rendering.get_latency())
    else:
        log.warning(f"Failed to log all metrics!")
    cv.destroyAllWindows()


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

    # args = build_argparser().parse_args()
    args = Args()
    args.model, args.config = model_opencv
    # args.model, args.config = model_openvino
    args.loop = False
    args.input = 'data/video/person-bicycle-car-detection.mp4'
    args.input = 'data/image/person-bicycle-car-detection'
    args.target = cv.dnn.DNN_TARGET_CPU
    args.prob_threshold = 0.65
    args.nms = 0.5
    args.mean = [0, 0, 0]
    args.scale = 1.0
    args.rgb = True
    if (args.model, args.config) == model_opencv:
        args.framework = 'darknet'
        args.backend = cv.dnn.DNN_BACKEND_OPENCV
        config = ConfigParser(strict=False)
        config.read(args.config)
        args.width, args.height = (int(config['net']['width']),
                                   int(config['net']['height']))
    else:
        args.framework = None
        args.backend = cv.dnn.DNN_BACKEND_INFERENCE_ENGINE
        config = etree.parse(args.config).getroot()
        # print(f"DEBUG: config = {config}")
        shape = config.find('layers/layer/data').attrib['shape'].split(',')
        # print(f"DEBUG: IE shape = {shape}")
        # return
        args.width, args.height = (int(shape[-1]), int(shape[-2]))
    path_coco_names = osp.abspath('coco.names')
    if osp.isfile(path_coco_names) and access(path_coco_names, R_OK):
        print(f"Class names = {path_coco_names}")
        args.classes = path_coco_names
    else:
        args.classes = None
    args.coi = [1, 2, 3]
    args.output = osp.join('results', osp.basename(args.input))
    args.no_show = False
    args.raw_output_message = True

    if args.output and isinstance(args.output, str):
        split = list(osp.splitext(args.output))
        if (args.model, args.config) == model_opencv:
            split.insert(1, f'.{OPENCV_MODEL}')
        else:
            split.insert(1, f'.{OPENVINO_MODEL}')
        args.output = ''.join(split)
        print(f"Saving to {args.output}...")
    else:
        print(f"Malformed output: {args.output}! Setting to 'None'")
        args.output = None

    cap = open_images_capture(osp.abspath(args.input), args.loop)

    try:
        from model_api.performance_metrics import PerformanceMetrics
    except ImportError:
        PerformanceMetrics = None
    try:
        from visualizers import ColorPalette
    except ImportError:
        ColorPalette = None

    inference_opencv(args, cap, PerformanceMetrics, ColorPalette)

    try:
        cap.cap.release()
    except AttributeError:
        try:
            cap.release()
        except:
            pass


backends = (
    cv.dnn.DNN_BACKEND_DEFAULT, cv.dnn.DNN_BACKEND_HALIDE,
    cv.dnn.DNN_BACKEND_INFERENCE_ENGINE, cv.dnn.DNN_BACKEND_OPENCV,
    cv.dnn.DNN_BACKEND_VKCOM, cv.dnn.DNN_BACKEND_CUDA
)

targets = (
    cv.dnn.DNN_TARGET_CPU, cv.dnn.DNN_TARGET_OPENCL,
    cv.dnn.DNN_TARGET_OPENCL_FP16, cv.dnn.DNN_TARGET_MYRIAD,
    cv.dnn.DNN_TARGET_HDDL, cv.dnn.DNN_TARGET_VULKAN,
    cv.dnn.DNN_TARGET_CUDA, cv.dnn.DNN_TARGET_CUDA_FP16
)

if __name__ == '__main__':
    from sys import stdout

    log.basicConfig(format='[ %(levelname)s ] %(message)s', level=log.DEBUG,
                    stream=stdout)
    main()

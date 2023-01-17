import sys

if sys.version_info[0] == 2:
    import Queue as queue
else:
    import queue

from os import environ, path as osp
from time import time
from threading import Thread

import cv2 as cv
import numpy as np


def add_argument(zoo, parser, name, help, required=False, default=None, type=None, action=None, nargs=None):
    if len(sys.argv) <= 1:
        return

    modelName = sys.argv[1]

    if osp.isfile(zoo):
        fs = cv.FileStorage(zoo, cv.FILE_STORAGE_READ)
        node = fs.getNode(modelName)
        if not node.empty():
            value = node.getNode(name)
            if not value.empty():
                if value.isReal():
                    default = value.real()
                elif value.isString():
                    default = value.string()
                elif value.isInt():
                    default = int(value.real())
                elif value.isSeq():
                    default = []
                    for i in range(value.size()):
                        v = value.at(i)
                        if v.isInt():
                            default.append(int(v.real()))
                        elif v.isReal():
                            default.append(v.real())
                        else:
                            print('Unexpected value format')
                            exit(0)
                else:
                    print('Unexpected field format')
                    exit(0)
                required = False

    if action == 'store_true':
        default = 1 if default == 'true' else (0 if default == 'false' else default)
        assert(default is None or default == 0 or default == 1)
        parser.add_argument('--' + name, required=required, help=help, default=bool(default),
                            action=action)
    else:
        parser.add_argument('--' + name, required=required, help=help, default=default,
                            action=action, nargs=nargs, type=type)


def add_preproc_args(zoo, parser, sample):
    aliases = []
    if osp.isfile(zoo):
        fs = cv.FileStorage(zoo, cv.FILE_STORAGE_READ)
        root = fs.root()
        for name in root.keys():
            model = root.getNode(name)
            if model.getNode('sample').string() == sample:
                aliases.append(name)

    parser.add_argument('alias', nargs='?', choices=aliases,
                        help='An alias name of model to extract preprocessing parameters from models.yml file.')
    add_argument(zoo, parser, 'model', required=True,
                 help='Path to a binary file of model contains trained weights. '
                      'It could be a file with extensions .caffemodel (Caffe), '
                      '.t7 or .net (Torch), .weights (Darknet), .bin (OpenVINO)')
    add_argument(zoo, parser, 'config',
                 help='Path to a text file of model contains network configuration. '
                      'It could be a file with extensions .prototxt (Caffe), .cfg (Darknet), .xml (OpenVINO)')
    add_argument(zoo, parser, 'mean', nargs='+', type=float, default=[0, 0, 0],
                 help='Preprocess input image by subtracting mean values. '
                      'Mean values should be in BGR order.')
    add_argument(zoo, parser, 'scale', type=float, default=1.0,
                 help='Preprocess input image by multiplying on a scale factor.')
    add_argument(zoo, parser, 'width', type=int,
                 help='Preprocess input image by resizing to a specific width.')
    add_argument(zoo, parser, 'height', type=int,
                 help='Preprocess input image by resizing to a specific height.')
    add_argument(zoo, parser, 'rgb', action='store_true',
                 help='Indicate that model works with RGB input images instead BGR ones.')
    add_argument(zoo, parser, 'classes',
                 help='Optional path to a text file with names of classes to label detected objects.')


def findFile(filename):
    if filename:
        if osp.exists(filename):
            return filename

        fpath = cv.samples.findFile(filename, False)
        if fpath:
            return fpath

        samplesDataDir = osp.join(osp.dirname(osp.abspath(__file__)),
                                      '..',
                                      'data',
                                      'dnn')
        if osp.exists(osp.join(samplesDataDir, filename)):
            return osp.join(samplesDataDir, filename)

        for path in ['OPENCV_DNN_TEST_DATA_PATH', 'OPENCV_TEST_DATA_PATH']:
            try:
                extraPath = environ[path]
                absPath = osp.join(extraPath, 'dnn', filename)
                if osp.exists(absPath):
                    return absPath
            except KeyError:
                pass

        print('File ' + filename + ' not found! Please specify a path to '
                                   '/opencv_extra/testdata in OPENCV_DNN_TEST_DATA_PATH environment '
                                   'variable or pass a full path to model.')
        exit(0)


class QueueFPS(queue.Queue):
    def __init__(self):
        queue.Queue.__init__(self)
        self.startTime = 0
        self.counter = 0

    def put(self, v):
        queue.Queue.put(self, v)
        self.counter += 1
        if self.counter == 1:
            self.startTime = time()

    def getFPS(self):
        return self.counter / (time() - self.startTime)


class WrapperDNN:
    def __init__(self, args, cap):
        self.args = args

        self.args.model = findFile(self.args.model)
        self.args.config = findFile(self.args.config)
        self.args.classes = findFile(self.args.classes)

        # Load names of classes
        self.classes = None
        if self.args.classes:
            with open(self.args.classes, 'rt') as f:
                self.classes = f.read().rstrip('\n').split('\n')

        self.confThreshold = self.args.prob_threshold
        self.nmsThreshold = self.args.nms

        self.net = None
        self.outNames = None

        self.cap = cap

        self.process = True

        self.framesQueue = None

        self.processedFramesQueue = None
        self.predictionsQueue = None

        self.winName = None

        self.inpWidth = None
        self.inpHeight = None

        if hasattr(cap, 'cap'):
            self.fps = cap.fps()
            cap = cap.cap
        elif hasattr(cap, 'get'):
            self.fps = cap.get(cv.CAP_PROP_FPS)
        else:
            self.fps = -1

        # print(f"DEBUG: {dir(cap)}")
        if args.output:
            self.out = cv.VideoWriter()
            self.outWidth = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
            self.outHeight = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
            self.out.open(args.output, cv.VideoWriter_fourcc(*'MJPG'),
                          self.fps, (self.outWidth, self.outHeight))
            # print(f"DEBUG: output shape = {(self.outWidth, self.outHeight)}...")
        else:
            self.out = None
            self.outWidth = None
            self.outHeight = None

        if hasattr(args, 'coi') and isinstance(args.coi, (dict, list, set, tuple)):
            self.classes_of_interest = np.array(tuple(args.coi), dtype=np.uint)
            self.classes_of_interest -= 1
        else:
            self.classes_of_interest = None

    def load_network(self):
        # Load a network
        self.net = cv.dnn.readNet(cv.samples.findFile(self.args.model),
                                  cv.samples.findFile(self.args.config),
                                  self.args.framework)
        self.net.setPreferableBackend(self.args.backend)
        self.net.setPreferableTarget(self.args.target)
        self.outNames = self.net.getUnconnectedOutLayersNames()
        # print(f"DEBUG: out names = {self.outNames}")

    def drawPred(self, frame, classId, conf, left, top, right, bottom):
        # if not all((0 <= left < self.inpWidth, 0 <= top < self.inpHeight,
        #             0 < right <= self.inpWidth, 0 < bottom <= self.inpHeight)):
        #     return None
        # Draw a bounding box
        # print(f"DEBUG: {classId, conf, left, top, right, bottom}")
        cv.rectangle(frame, (left, top), (right, bottom), (0, 255, 0))

        label = '%.2f' % conf

        # Print a label of class
        if self.classes:
            assert (classId < len(self.classes))
            label = '%s: %s' % (self.classes[classId], label)

        labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(top, labelSize[1])
        cv.rectangle(frame, (left, top - labelSize[1]), (left + labelSize[0], top + baseLine), (255, 255, 255),
                     cv.FILLED)
        cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

    def postprocess(self, frame, outs):
        frameHeight = frame.shape[0]
        frameWidth = frame.shape[1]
        # print(f"DEBUG: post-frame dimensions = {frameWidth, frameHeight}")

        layerNames = self.net.getLayerNames()
        lastLayerId = self.net.getLayerId(layerNames[-1])
        lastLayer = self.net.getLayer(lastLayerId)

        confThreshold = self.confThreshold
        nmsThreshold = self.nmsThreshold
        coi = self.classes_of_interest

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
                    if confidence > self.confThreshold:
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
        if len(self.outNames) > 1 or lastLayer.type == 'Region' and self.args.backend != cv.dnn.DNN_BACKEND_OPENCV:
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
            self.drawPred(frame, classIds[i], confidences[i], left, top,
                          left + width, top + height)

    # Process inputs
    def init_gui(self):
        self.winName = 'Deep learning object detection in OpenCV'
        cv.namedWindow(self.winName, cv.WINDOW_NORMAL)

        # cv.createTrackbar('Confidence threshold, %', self.winName,
        #                   int(self.confThreshold * 100), 99, self.callback)

    def callback(self, pos):
        self.confThreshold = pos / 100.0

    def inference(self):
        self.process = True

        if self.net is None:
            self.load_network()

        self.framesQueue = QueueFPS()

        self.processedFramesQueue = queue.Queue()
        self.predictionsQueue = QueueFPS()

        framesThread = Thread(target=self.framesThreadBody)
        framesThread.start()

        processingThread = Thread(target=self.processingThreadBody)
        processingThread.start()

        # Postprocessing and rendering loop
        while True:  # cv.waitKey(1) < 0:
            try:
                # Request prediction first because they put after frames
                outs = self.predictionsQueue.get_nowait()
                # outs = self.predictionsQueue.get()
                frame = self.processedFramesQueue.get_nowait()
                # frame = self.processedFramesQueue.get()

                self.postprocess(frame, outs)

                # Put efficiency information
                if True and self.predictionsQueue.counter > 1:
                    label = 'Camera: %.2f FPS' % (self.framesQueue.getFPS())
                    cv.putText(frame, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))

                    label = 'Network: %.2f FPS' % (self.predictionsQueue.getFPS())
                    cv.putText(frame, label, (0, 30), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))

                    label = 'Skipped frames: %d' % (self.framesQueue.counter - self.predictionsQueue.counter)
                    cv.putText(frame, label, (0, 45), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))

                if self.out is not None and self.out.isOpened():
                    self.out.write(cv.resize(frame, (self.outWidth, self.outHeight), cv.INTER_NEAREST))

                if self.winName:
                    cv.imshow(self.winName, frame)
                    key = cv.waitKey(1)

                    ESC_KEY = 27
                    # Quit
                    if key in {ord('q'), ord('Q'), ESC_KEY}:
                        break

                # Terminate if input queue is empty
                if self.framesQueue.empty():
                    break
                # else:
                #     continue
            except queue.Empty:
                pass

        self.process = False

        framesThread.join()
        processingThread.join()

        if self.out is not None:
            self.out.release()

    # Frames capturing thread
    def framesThreadBody(self):
        while self.process:
            hasFrame, frame = self.cap.read()
            # frame = self.cap.read()
            if not hasFrame:
            # if frame is None:
                break
            self.framesQueue.put(frame)

    # Frames processing thread
    def processingThreadBody(self):
        futureOutputs = []
        while self.process:
            # Get next frame
            frame = None
            try:
                frame = self.framesQueue.get_nowait()
                # frame = self.framesQueue.get()

                if self.args.asyncN:
                    if len(futureOutputs) == self.args.asyncN:
                        frame = None  # skip the frame
                else:
                    self.framesQueue.queue.clear()  # skip the rest of frames
            except queue.Empty:
                pass

            if frame is not None:
                frameHeight = frame.shape[0]
                frameWidth = frame.shape[1]

                # Create a 4D blob from a frame.
                self.inpWidth = self.args.width if self.args.width else frameWidth
                self.inpHeight = self.args.height if self.args.height else frameHeight
                # print(f"DEBUG: args dims = {self.args.width, self.args.height}")
                frame = cv.resize(frame, (self.inpWidth, self.inpHeight))
                blob = cv.dnn.blobFromImage(
                    frame, size=(self.inpWidth, self.inpHeight),
                    swapRB=self.args.rgb, ddepth=cv.CV_8U
                )
                self.processedFramesQueue.put(frame)

                # Run a model
                self.net.setInput(blob, scalefactor=self.args.scale, mean=self.args.mean)
                if self.net.getLayer(0).outputNameToIndex('im_info') != -1:  # Faster-RCNN or R-FCN
                    self.net.setInput(
                        np.array([[self.inpHeight, self.inpWidth, 1.6]],
                                 dtype=np.float32),
                        'im_info'
                    )

                if self.args.asyncN:
                    futureOutputs.append(self.net.forwardAsync())
                else:
                    outs = self.net.forward(self.outNames)
                    self.predictionsQueue.put(np.copy(outs))

            while futureOutputs and futureOutputs[0].wait_for(0):
                out = futureOutputs[0].get()
                self.predictionsQueue.put(np.copy([out]))

                del futureOutputs[0]


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
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

    parser = ArgumentParser(add_help=False)
    parser.add_argument('--zoo', default=osp.join(osp.dirname(osp.abspath(__file__)), 'models.yml'),
                        help='An optional path to file with preprocessing parameters.')
    parser.add_argument('--input',
                        help='Path to input image or video file. Skip this argument to capture frames from a camera.')
    parser.add_argument('--framework', choices=['caffe', 'torch', 'darknet', 'dldt'],
                        help='Optional name of an origin framework of the model. '
                             'Detect it automatically if it does not set.')
    # parser.add_argument('--thr', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('-t', '--prob_threshold', default=0.5, type=float,
                        help='Optional. Probability threshold for detections filtering.')
    parser.add_argument('--nms', type=float, default=0.4, help='Non-maximum suppression threshold')
    parser.add_argument('--backend', choices=backends, default=cv.dnn.DNN_BACKEND_DEFAULT, type=int,
                        help="Choose one of computation backends: "
                             "%d: automatically (by default), "
                             "%d: Halide language (http://halide-lang.org/), "
                             "%d: Intel's Deep Learning Inference Engine (https://software.intel.com/openvino-toolkit), "
                             "%d: OpenCV implementation, "
                             "%d: VKCOM, "
                             "%d: CUDA" % backends)
    parser.add_argument('--target', choices=targets, default=cv.dnn.DNN_TARGET_CPU, type=int,
                        help='Choose one of target computation devices: '
                             '%d: CPU target (by default), '
                             '%d: OpenCL, '
                             '%d: OpenCL fp16 (half-float precision), '
                             '%d: NCS2 VPU, '
                             '%d: HDDL VPU, '
                             '%d: Vulkan, '
                             '%d: CUDA, '
                             '%d: CUDA fp16 (half-float preprocess)' % targets)
    parser.add_argument('--async', type=int, default=0,
                        dest='asyncN',
                        help='Number of asynchronous forwards at the same time. '
                             'Choose 0 for synchronous mode')
    parser.add_argument('-o', '--output', required=False,
                        help='Optional. Name of the output file(s) to save.')
    args, _ = parser.parse_known_args()

    add_preproc_args(args.zoo, parser, 'object_detection')
    parser = ArgumentParser(parents=[parser],
                            description='Use this script to run object detection deep learning networks using OpenCV.',
                            formatter_class=ArgumentDefaultsHelpFormatter)
    args = parser.parse_args()

    cap = cv.VideoCapture(cv.samples.findFileOrKeep(args.input) if args.input else 0)
    pipeline = WrapperDNN(args, cap)
    pipeline.init_gui()
    pipeline.inference()

"""
Classification sample

Command line to run:
python ie_classification_sample.py -i image.jpg \
    -m squeezenet1.1.xml -w squeezenet1.1.bin -c imagenet_synset_words.txt
"""

import os
import cv2
import sys
import argparse
import numpy as np
import logging as log
from openvino.inference_engine import IECore


class InferenceEngineClassifier:
    def __init__(self, configPath=None, weightsPath=None,
            device='CPU', classesPath=None):
        self.weights = weightsPath
        self.config = configPath
        self.ie = IECore()
        self.net = self.ie.read_network(model=configPath, weights=weightsPath)
        self.exec_net = self.ie.load_network(network=self.net,
                                             device_name=device)
        if classesPath:
            self.classes = [line.rstrip('\n') for line in open(classesPath)]
        return

    def get_top(self, prob, topN=1):
        prob = np.squeeze(prob)
        best_n = np.argsort(prob)[-topN:]
        result = []
        for i in range(len(best_n)-1, -1, -1):
            try:
                classname = self.classes[int(best_n[i])]
            except:
                classname = best_n[i]
            line = [classname, prob[best_n[i]] * 100.0]
            result.append(line)
        return result

    def _prepare_image(self, image, h, w):
        if image.shape[:-1] != (h, w):
            image = cv2.resize(image, (w, h))
        image = image.transpose((2, 0, 1))
        return image

    def classify(self, image):
        input_blob = next(iter(self.net.input_info))
        out_blob = next(iter(self.net.outputs))
        n, c, h, w = self.net.input_info[input_blob].input_data.shape
        blob = self._prepare_image(image, h, w)
        output = self.exec_net.infer(inputs={input_blob: blob})
        return output[out_blob]


def build_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', help='Path to an .xml \
        file with a trained model.', required=True, type=str)
    parser.add_argument('-w', '--weights', help='Path to an .bin file \
        with a trained weights.', required=True, type=str)
    parser.add_argument('-i', '--input', help='Path to \
        image file', required=True, type=str)
    parser.add_argument('-d', '--device', help='Specify the target \
        device to infer on; CPU, GPU, FPGA or MYRIAD is acceptable. \
        Sample will look for a suitable plugin for device specified \
        (CPU by default)', default='CPU', type=str)
    parser.add_argument('-c', '--classes', help='File containing classes \
        names', type=str, default=None)
    return parser


def main():
    log.basicConfig(format="[ %(levelname)s ] %(message)s",
        level=log.INFO, stream=sys.stdout)
    args = build_argparser().parse_args()

    log.info("Start IE classification sample")

    ie_classifier = InferenceEngineClassifier(configPath=args.model,
        weightsPath=args.weights, device=args.device, classesPath=args.classes)

    img = cv2.imread(args.input)

    prob = ie_classifier.classify(img)
    predictions = ie_classifier.get_top(prob, 5)
    log.info("Predictions: " + str(predictions))

    return


if __name__ == '__main__':
    sys.exit(main())

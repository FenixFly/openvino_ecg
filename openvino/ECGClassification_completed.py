"""
ECG classifier based on Inference Engine

"""

import os
import cv2
import sys
import argparse
import numpy as np
import pandas as pd
import logging as log
from openvino.inference_engine import IECore

class ECGClassifier:
    def __init__(self, configPath=None, weightsPath=None,
            device='CPU'):
        self.ie = IECore()
        self.net = self.ie.read_network(model=configPath, weights=weightsPath)
        self.exec_net = self.ie.load_network(network=self.net, device_name=device)

    def _prepare_data(self, df, n, l):
        data = df.astype(np.float)
        if data.shape[-1] != l:
            raise RuntimeError('Input data is not acceptable to model')
        return data

    def classify(self, data):
    
        input_blob = next(iter(self.net.input_info))
        out_blob = next(iter(self.net.outputs))
        n, l = self.net.input_info[input_blob].input_data.shape
        blob = self._prepare_data(data, n, l)
        output = self.exec_net.infer(inputs = {input_blob: blob})
        output = output[out_blob]
        return output


def build_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', help='Path to an .xml \
        file with a trained model.', required=True, type=str)
    parser.add_argument('-w', '--weights', help='Path to an .bin file \
        with a trained weights.', required=True, type=str)
    parser.add_argument('-i', '--input', help='Path to \
        input data', required=True, type=str)
    parser.add_argument('-d', '--device', help='Specify the target \
        device to infer on; CPU, GPU, FPGA or MYRIAD is acceptable. \
        Sample will look for a suitable plugin for device specified \
        (CPU by default)', default='CPU', type=str)
    return parser

    
def main():
    log.basicConfig(format="[ %(levelname)s ] %(message)s",
        level=log.INFO, stream=sys.stdout)
    args = build_argparser().parse_args()

    log.info("Start ECG classification sample")
    
    # Create ECGClassifier
    ie_classifier = ECGClassifier(configPath=args.model,
        weightsPath=args.weights, device=args.device)

    # Read data with pandas
    data = pd.read_csv(args.input, header=None, na_values="?")
    
    # Start inference
    prob = ie_classifier.classify(data)
    
    # Show prediciton
    log.info("Predictions: " + str(prob))
    
    return


if __name__ == '__main__':
    sys.exit(main())

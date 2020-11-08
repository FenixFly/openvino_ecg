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

        pass

    def _prepare_data(self, df, n, l):

        pass

    def classify(self, data):
    
        pass


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

    # Read data with pandas

    # Start inference

    # Show prediciton

    return


if __name__ == '__main__':
    sys.exit(main())

"""
ECG segmenter based on Inference Engine

"""

import os
import cv2
import sys
import argparse
import numpy as np
import pandas as pd
import logging as log
from openvino.inference_engine import IECore
import matplotlib.pyplot as plt

class ECGSegmenter:
    def __init__(self, configPath=None, weightsPath=None,
            device='CPU'):
        self.ie = IECore()
        self.net = self.ie.read_network(model=configPath, weights=weightsPath)
        self.exec_net = self.ie.load_network(network=self.net, device_name=device)


    def _prepare_data(self, df, k, n, l):
        data = df.astype(np.float)
        if data.shape[-1] != l:
            raise RuntimeError('Input data is not acceptable to model') 
        return data


    def segment(self, data):
        
        input_blob = next(iter(self.net.input_info))
        out_blob = next(iter(self.net.outputs))

        k, n, l = self.net.input_info[input_blob].input_data.shape
        blob = self._prepare_data(data,k,n,l)
        output = self.exec_net.infer(inputs = {input_blob: blob})
        output = output[out_blob]

        return output

    def process_output(self, data, results):
    
        # Code for plot
        v_to_del = {1:'p', 2:'qrs', 3:'t'}
        sample_rate = 500

        def remove_small(signal):
            max_dist = 12
            last_zero = 0
            for i in range(len(signal)):
                if signal[i] == 0:
                    if i - last_zero < max_dist:
                        signal[last_zero:i] = 0
                    last_zero = i

        def merge_small(signal):
            max_dist = 12
            lasts = np.full(signal.max() + 1, -(max_dist+1))
            for i in range(len(signal)):
                m = signal[i]
                if i - lasts[m] < max_dist and m > 0:
                    signal[lasts[m]:i] = m
                lasts[m] = i

        def mask_to_delineation(mask):
            merge_small(mask)
            remove_small(mask)
            delineation = {'p':[], 'qrs':[], 't':[]}
            i = 0
            mask_length = len(mask)
            while i < mask_length:
                v = mask[i]
                if v > 0:
                    delineation[v_to_del[v]].append([i, 0])
                    while i < mask_length and mask[i] == v:
                        delineation[v_to_del[v]][-1][1] = i
                        i += 1
                    t = delineation[v_to_del[v]][-1]
                i += 1
            return delineation
                
        wave_type_to_color = {
                    "p": "yellow",
                    "qrs": "red",
                    "t": "green"
                    }

        def plot_signal_with_mask(signal, mask):
            plt.figure(figsize=(18, 5))
            plt.title("Сигнал с маской")
            plt.xlabel("Время (сек)")
            plt.ylabel("Амплитуда (мВ)")
            x_axis_values = np.linspace(0, len(signal) / sample_rate, len(signal))
            plt.plot(x_axis_values, signal, linewidth=2, color="black")
            
            delineation = mask_to_delineation(mask)
            for wave_type in ["p", "qrs", "t"]:
                color = wave_type_to_color[wave_type]
                for begin, end in delineation[wave_type]:
                    begin /= sample_rate
                    end /= sample_rate
                    plt.axvspan(begin, end, facecolor=color, alpha=0.5)

        # Process output
        results = np.squeeze(results)
        results = results.argmax(axis=0)
        results[:500] = 0
        results[-500:] = 0

        # Make plot
        plot_signal_with_mask(data, results)
        plt.savefig('plot.png')




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

    log.info("Start ECG segmentation sample")
    
    # Create ECGSegmenter
    ie_segmenter = ECGSegmenter(configPath=args.model,
        weightsPath=args.weights, device=args.device)

    # Read data with numpy
    data = np.loadtxt(args.input)

    # Start inference
    segmentation = ie_segmenter.segment(data)

    # Show segmentation
    ie_segmenter.process_output(data, segmentation)

    return


if __name__ == '__main__':
    sys.exit(main())

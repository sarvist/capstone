#Brainflow and EEG Imports
import logging
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, WindowOperations, DetrendOperations
from brainflow.ml_model import MLModel, BrainFlowMetrics, BrainFlowClassifiers, BrainFlowModelParams
from sklearn.cross_decomposition import CCA
from pyqtgraph.Qt import QtGui, QtCore

#General Imports
import numpy as np
import timeit
import time
from threading import Timer

#GUI Imports
import json
import websocket

#Eye Tracking Imports
import ml_eye_tracker

#Radio Imports
import RPi.GPIO as GPIO  
import spidev
from lib_nrf24 import NRF24







def main():
    BoardShim.enable_dev_board_logger()
    params = BrainFlowInputParams()
    try:
        board = BoardShim(BoardIds.SYNTHETIC_BOARD, params)
        board.prepare_session()
        board.start_stream(450000, '')
        DataAcquisition(board)
    except BaseException:
        logging.warning('Exception', exc_info=True)
    finally:
        logging.info('End')
        if board.is_prepared():
            logging.info('Releasing session')
            board.release_session()

class RepeatedTimer(object):
    def __init__(self, interval, function, fn2, *args, **kwargs):
        self._timer = None
        self.interval = interval
        self.function = function
        self.fn2 = fn2
        self.args = args
        self.kwargs = kwargs
        self.is_running = False
        self.start()

    def _run(self):
        self.is_running = False
        self.start()
        self.function(*self.args, **self.kwargs)
        self.fn2(*self.args, **self.kwargs)

    def start(self):
        if not self.is_running:
            self._timer = Timer(self.interval, self._run)
            self._timer.start()
            self.is_running = True

    def stop(self):
        self._timer.cancel()
        self.is_running = False

if __name__ == "__main__":
    main()
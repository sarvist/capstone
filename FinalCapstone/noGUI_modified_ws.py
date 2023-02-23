import argparse
import time
import logging
from threading import Timer
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds, BrainFlowPresets
from brainflow.data_filter import DataFilter, FilterTypes, WindowOperations, DetrendOperations, AggOperations
import json
import websocket
import time
import numpy as np

ws = websocket.WebSocket()

ws.connect('ws://localhost:8000/ws/polData/')


class DataAcquisition:
    def __init__(self, board):
        self.board = board
        self.board_id = board.get_board_id()
        self.exg_channels = BoardShim.get_exg_channels(self.board_id)
        self.sampling_rate = BoardShim.get_sampling_rate(self.board_id)
        self.window_size = 4
        self.num_points = self.window_size * self.sampling_rate
        self.cycle = 0
        self.isPrimed = 0
        self.blinkCounter = 0
        self.time_int = 0.05
        RepeatedTimer(self.time_int, self.update, self.p)
        while True:
            pass  # takes like 3-5 seconds after stop to give up ganglion

    def update(self):
        data = self.board.get_current_board_data(self.num_points)
        avg_bands = self.filterData(data)
        self.checkForBlinks(avg_bands)
        # downsampled_data = DataFilter.perform_downsampling(data[1], 3, AggOperations.MEDIAN.value)
        # print(len(downsampled_data))
        # create json string to send via websocket
        json_string = json.dumps({"isPrimed": self.isPrimed, "blinks": self.blinkCounter,  "ch1": data[0].tolist(),
                                  "ch2": data[1].tolist(), "ch3": data[2].tolist(), "ch4": data[3].tolist(),
                                  "b0": avg_bands[0], "b1": avg_bands[1], "b2": avg_bands[2], "b3": avg_bands[3], "b4": avg_bands[4]});
        ws.send(json_string)

        # Calculate the size of the JSON string in bytes
        data_size = len(json_string.encode())
 
        # Calculate the data rate in bits per second 0.05s interval used
        data_rate = data_size * 8 / self.time_int

        # Convert the data rate to kilobits per second (Kbps) or megabits per second (Mbps)
        data_rate_kbps = data_rate / 1000
        data_rate_mbps = data_rate_kbps / 1000
        print("\nData rate in mbps: " + str(data_rate_mbps));

    def p(self):
        pass

    def checkForBlinks(self, avg_bands):
        if ((avg_bands[1]/1) < 8):  # 2 channel
            self.cycle += 1
            if (self.cycle > 20):
                # print('\r System Status: Primed - Number of blinks: ' +
                #     str(self.blinkCounter), end='')
                self.isPrimed = 1
        else:
            self.cycle = 0
            # print('\r System Status: Not    - Number of blinks: ' +
            #     str(self.blinkCounter), end='')
            if (self.isPrimed == 1):
                self.blinkCounter += 1
                '''
                This is where Sam should call the eye tracking program
                '''
                self.isPrimed = 0

    def filterData(self, data):
        avg_bands = [0, 0, 0, 0, 0]
        counter = 0
        for count, channel in enumerate(self.exg_channels):
            DataFilter.detrend(data[channel], DetrendOperations.CONSTANT.value)
            DataFilter.perform_bandpass(data[channel], self.sampling_rate, 3.0, 45.0, 2,
                                        FilterTypes.BUTTERWORTH.value, 0)
            DataFilter.perform_bandstop(data[channel], self.sampling_rate, 48.0, 52.0, 2,
                                        FilterTypes.BUTTERWORTH.value, 0)
            DataFilter.perform_bandstop(data[channel], self.sampling_rate, 58.0, 62.0, 2,
                                        FilterTypes.BUTTERWORTH.value, 0)
            self.psd_size = DataFilter.get_nearest_power_of_two(
                self.sampling_rate)
            if data.shape[1] > self.psd_size:
                psd_data = DataFilter.get_psd_welch(data[channel], self.psd_size, self.psd_size // 2,
                                                    self.sampling_rate,
                                                    WindowOperations.BLACKMAN_HARRIS.value)
                avg_bands[0] = avg_bands[0] + \
                    DataFilter.get_band_power(psd_data, 2.0, 4.0)
                avg_bands[1] = avg_bands[1] + \
                    DataFilter.get_band_power(psd_data, 4.0, 8.0)
                avg_bands[2] = avg_bands[2] + \
                    DataFilter.get_band_power(psd_data, 8.0, 13.0)
                avg_bands[3] = avg_bands[3] + \
                    DataFilter.get_band_power(psd_data, 13.0, 30.0)
                avg_bands[4] = avg_bands[4] + \
                    DataFilter.get_band_power(psd_data, 30.0, 50.0)

            counter += 1  # Does the first two
            if (counter == 1):
                break  # i know this is bad whatever
        return avg_bands


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

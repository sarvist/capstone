#Brainflow and EEG Imports
import logging
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, WindowOperations, DetrendOperations
from sklearn.cross_decomposition import CCA
from pyqtgraph.Qt import QtGui, QtCore #Nundita will remove these
import pyqtgraph as pg #Nundita will remove these

#General Imports
import numpy
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

class Graph:
    def __init__(self, board_shim):
        self.board_id = board_shim.get_board_id()
        self.board_shim = board_shim
        self.exg_channels = BoardShim.get_exg_channels(self.board_id)
        self.sampling_rate = BoardShim.get_sampling_rate(self.board_id)
        self.update_speed_ms = 50
        self.window_size = 4
        self.num_points = self.window_size * self.sampling_rate

        self.app = QtGui.QApplication([])
        self.win = pg.GraphicsWindow(title='BrainFlow Plot', size=(900, 800))

        #Inits
        self._init_timeseries()
        self._init_labels()
        self._init_radio()
        
        # Get each one of our reference signals for CCA
        # 800 Samples takes about 4 seconds to start
        
        self.freq1=self.getReferenceSignals(800,10) # number of samples, target freq
        self.freq2=self.getReferenceSignals(800,11)
        self.freq3=self.getReferenceSignals(800,12)
        self.freq4=self.getReferenceSignals(800,14)
        
        #Init Variables
        self.cycle = 0
        self.isPrimed = 0
        self.blinkCounter = 0
        self.activeClass = 4
        self.countSS = 0
        self.start = 0
        self.end = 0
        
        
        #Timer
        timer = QtCore.QTimer()
        timer.timeout.connect(self.update)
        timer.start(self.update_speed_ms)
        QtGui.QApplication.instance().exec_()
        
    '''
    Inits for the radio communication
    Ian will update this a little bit
    '''
    def _init_radio(self):
        GPIO.setmode(GPIO.BCM)
        radio = NRF24(GPIO, spidev.SpiDev())
        pipes = [[0xE0, 0xE0, 0xF1, 0xF1, 0xE1],[0xF1, 0xF1, 0xF0, 0xF0, 0xE0]]
        radio.begin(0, 25)
        radio.setPayloadSize(2)
        radio.setPALevel(NRF24.PA_MIN)
        radio.setAutoAck(False)
        radio.openWritingPipe(pipes[0])
        radio.stopListening()

    '''
    Nundita will remove this  
    '''
    def _init_timeseries(self):
        self.plots = list()
        self.curves = list()
        for i in range(len(self.exg_channels)):
            p = self.win.addPlot(row=i, col=0)
            p.showAxis('left', False)
            p.setMenuEnabled('left', False)
            p.showAxis('bottom', False)
            p.setMenuEnabled('bottom', False)
            if i == 0:
                p.setTitle('TimeSeries Plot')
            self.plots.append(p)
            curve = p.plot()
            self.curves.append(curve)
            
    def _init_labels(self):
        self.thetaScore = self.win.addLabel('theta:',row=3, col=0)
        self.textPlotg5 = self.win.addLabel('Scores 10:',row=4, col=0)
        self.textPlotg8 = self.win.addLabel('Scores 11:',row=5, col=0)
        self.textPlotg11 = self.win.addLabel('Scores 12:',row=6, col=0)
        self.textPlotg14 = self.win.addLabel('Scores: 14',row=7, col=0)
        self.textPlotg = self.win.addLabel('Not Currently Viewing: ',row=8, col=0)
        self.textPlotg15 = self.win.addLabel('Eye direction:',row=9, col=0)
        
    def update(self):
        data = self.board_shim.get_current_board_data(self.num_points)
        avg_bands = [0, 0, 0, 0, 0]
        for count, channel in enumerate(self.exg_channels):
            DataFilter.detrend(data[channel], DetrendOperations.CONSTANT.value)
            DataFilter.perform_bandpass(data[channel], self.sampling_rate, 3.0, 45.0, 2,
                                        FilterTypes.BUTTERWORTH.value, 0)
            DataFilter.perform_bandstop(data[channel], self.sampling_rate, 48.0, 52.0, 2,
                                        FilterTypes.BUTTERWORTH.value, 0)
            DataFilter.perform_bandstop(data[channel], self.sampling_rate, 58.0, 62.0, 2,
                                        FilterTypes.BUTTERWORTH.value, 0)
            self.curves[count].setData(data[channel].tolist())
            
            self.psd_size = DataFilter.get_nearest_power_of_two(self.sampling_rate)
            if data.shape[1] > self.psd_size:
                psd_data = DataFilter.get_psd_welch(data[channel], self.psd_size, self.psd_size // 2,
                                                        self.sampling_rate,
                                                        WindowOperations.BLACKMAN_HARRIS.value)
                #Theta Band power
                avg_bands[1] = avg_bands[1] + DataFilter.get_band_power(psd_data, 4.0, 8.0)
            
            #First electrode gets our theta value to tell if the system is ready for use
            if(count == 0): self.thetaScore.setText('theta: ' + str(avg_bands[1]))
            
            #Second electrode that is connected to O1 or O2
            if(count == 1):
                
                #Doing the CCA
                n_components=1
                freq=numpy.array([self.freq1,self.freq2,self.freq3,self.freq4])
                x = data[channel]
                y = numpy.array(x).reshape((1,800))            
                result = self.findCorr(n_components,y,freq)
                #max_result = max(result,key=float)
                #predictedClass = numpy.argmax(result)+1
                
                #Thresholds for SSVEP Classification
                countsToBeHad = 9
                
                if(self.activeClass == 4):
                    self.activeClass = self.getMax(result)
                    if(self.activeClass == 4):
                        #self.sendStuff('s')
                        pass
                    else:
                        self.countSS +=1
                else:
                    if(self.getMax(result) == self.activeClass):
                        self.countSS +=1
                        if(self.countSS > countsToBeHad):
                            predict = ''
                            if(self.activeClass == 0): predict = '10 HZ'
                            elif(self.activeClass == 1):predict = '11 HZ'
                            elif(self.activeClass == 2):predict = '12 HZ'
                            elif(self.activeClass == 3):predict = '14 HZ'
                            self.textPlotg.setText('Currently Viewing:' + predict)
                            
                
                            '''
                            assignemtns 10 and 12hz left, 11 and 14 hz right
                            
                            '''
                            
                            #Gets the eye tracking results in a dict
                            preds = ml_eye_tracker.track_eyes(0)
                            resultLeft = preds[0]['left eye']
                            resultRight = preds[-1]['right eye']
                            print(preds)
                            if(resultLeft == resultRight): # If both looking the same direction
                                self.textPlotg15.setText('Currently eyes are:' + resultLeft)

                                if(self.activeClass == 0 and (resultLeft == 'Left' or resultRight == 'Left')):
                                    self.sendStuff('f')
                                elif(self.activeClass == 2 and (resultLeft == 'Left' or resultRight == 'Left')):
                                    self.sendStuff('b')
                                elif(self.activeClass == 1 and (resultLeft == 'Right' or resultRight == 'Right')):
                                    self.sendStuff('l')
                                elif(self.activeClass == 3 and (resultLeft == 'Right' or resultRight == 'Right')):
                                    self.sendStuff('r')
                            #else: self.sendStuff('s')
                               
                    else:# reset
                        self.activeClass = 4
                        self.countSS = 0
                        self.textPlotg.setText('Not Currently Viewing:')
                        #self.sendStuff('s')
                
                #Displaying the new values of CCA 
                self.textPlotg5.setText('Value 10: ' + str(result[0]))
                self.textPlotg8.setText('Value 11: ' + str(result[1]))
                self.textPlotg11.setText('Value 12: ' + str(result[2]))
                self.textPlotg14.setText('Value 14: ' + str(result[3]))
                break
        self.app.processEvents()
         
    '''
    Reference Signals used for CCA
    at 200hz since ganglion is near there(maybe a little less due to bluetooth)
    '''
    def getReferenceSignals(self,length,target_freq):
        reference_signals = []
        t = numpy.arange(0, (length/(200)), step=1.0/(200))
        reference_signals.append(numpy.sin(numpy.pi*2*target_freq*t))
        reference_signals.append(numpy.cos(numpy.pi*2*target_freq*t))
        reference_signals.append(numpy.sin(numpy.pi*4*target_freq*t))
        reference_signals.append(numpy.cos(numpy.pi*4*target_freq*t))
        reference_signals = numpy.array(reference_signals)
        return reference_signals
                       
    def getMax(self, result):
        predictedClass = numpy.argmax(result)
        max_result = max(result,key=float)
        counter = 0
        tolerance = 0
        thres = 0.14
                       
        for i in range(len(result)):
            if(i != predictedClass):
                counter+=result[i]
        
        if((max_result > ((counter/3) + tolerance)) and max_result > thres):
           return predictedClass
        else:
            return 4
            
                       
    
    '''
    Method that sends to the arduino
    '''
    def sendStuff(self,message):
        
        if(message == 'f'): #Forward command is one
            message = str('F')
        elif(message == 's'):#stop command is two
            message = str('S')
        elif(message == 'l'):# Left command is three
            message = str('L')
        elif(message == 'r'): #Right command is 4
            message = str('R')
        elif(message == 'b'): #backwards command is 5
            message = str('B')
            
        sendMessage = list(message)  #the message to be sent   
        sendMessage.append(0)
        
        self.radio.write(sendMessage)   # just write the message to radio
    
    '''
    CCA
    '''
    def findCorr(self,n_components,numpyBuffer,freq):
        cca = CCA(n_components)
        corr=numpy.zeros(n_components)
        result=numpy.zeros((freq.shape)[0])
        for freqIdx in range(0,(freq.shape)[0]):
            cca.fit(numpyBuffer.T,numpy.squeeze(freq[freqIdx,:,:]).T)
            O1_a,O1_b = cca.transform(numpyBuffer.T, numpy.squeeze(freq[freqIdx,:,:]).T)
            indVal=0
            for indVal in range(0,n_components):
                corr[indVal] = numpy.corrcoef(O1_a[:,indVal],O1_b[:,indVal])[0,1]
            result[freqIdx] = numpy.max(corr)
        return result
        
def main():
    BoardShim.enable_dev_board_logger()
    logging.basicConfig(level=logging.DEBUG)

    params = BrainFlowInputParams()
    params.ip_port = 0
    params.serial_port = '/dev/ttyACM0'
    params.mac_address = ''
    params.other_info = ''
    params.serial_number = ''
    params.ip_address = ''
    params.ip_protocol = 0
    params.timeout = 20
    params.file = ''
    params.master_board = BoardIds.NO_BOARD
    try:
        board = BoardShim(BoardIds.GANGLION_BOARD.value,params)
        board.prepare_session()
        board.start_stream(450000, '')
        Graph(board)
    except BaseException:
        logging.warning('Exception', exc_info=True)
    finally:
        logging.info('End')
        if board.is_prepared():
            logging.info('Releasing session')
            board.release_session()


if __name__ == '__main__':
    main()


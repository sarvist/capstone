'''
Ian Lead here

'''


import threading

class Transmission(threading.Thread):
    def __init__(self,thread_name,thread_ID, scores):
        threading.Thread.__init__(self, daemon=True) 
        self.thread_name = thread_name 
        self.thread_ID = thread_ID 
        self.scores = scores
        self.freqs = [5,7,8,9]

    def run(self):
        commandToSend = self.interpretScores()
        print(commandToSend)


    
    def transmit():
        pass



    def interpretScores(self):
        max_value = max(self.scores)
        if(max_value < 0.1):
            max_index = 5 #meaning no command
        else:
            max_index = self.scores.index(max_value)

        #i would use matching, but dont think rpi has 3.10
        if(max_index == 0):
            return 'Left'
        elif(max_index == 1):
            return 'Right'
        elif(max_index == 2):
            return 'forward'
        elif(max_index == 3):
            return 'backwards'
        elif(max_index == 4):
            return 'Stop'




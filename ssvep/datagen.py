import threading
import random
import time
from queue import Queue

class DataGen(threading.Thread):
    def __init__(self,thread_name, thread_ID):
        threading.Thread.__init__(self, daemon=True) 
        self.thread_name = thread_name 
        self.thread_ID = thread_ID 
        self.q = Queue(maxsize=5)

    def run(self):
        while(1):
            voltage = random.randint(0,1)
            voltage = 2*(voltage-0.5)
            self.q.put(voltage)
            time.sleep(1/240)
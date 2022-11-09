import datagen
import threading
import time
import analysis
import transmission


OFFSET = 500

def main():
    cca = analysis.Analysis()
    thread1 = datagen.DataGen('thread1', 1)
    thread1.start()
    data = []
    prevLength = 0

    while(1):
        data.append(thread1.q.get())
        if(len(data) > prevLength + OFFSET):
            #filters here on the data
            scores = cca.performCCA(data[prevLength:prevLength+OFFSET])
            prevLength = len(data)
            thread = transmission.Transmission('threadTrans', 2, scores)
            thread.start()
            print(scores)

if __name__ == '__main__':
    main()
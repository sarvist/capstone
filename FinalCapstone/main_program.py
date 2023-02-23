import ml_eye_tracker
import time


fid = 0
while True:
    preds = ml_eye_tracker.track_eyes(fid)
    if len(preds) > 0:
        print(preds)
    fid += 1

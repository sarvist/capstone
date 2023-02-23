import numpy as np
import cv2 as cv
from tflite_runtime.interpreter import Interpreter
# import tensorflow as tf
# import custom_loss

model_w, model_h, model_c = 47, 23, 1

# model filename
model_version = 10
model_type = 'tflite'
char_to_int = {'l': 0, 'r': 1, 'c': 2, '?': 3}
pred_to_name = {0: 'Left', 1: 'Right', 2: 'Center', 3: 'Unsure'}

# number of frames to skip before running prediction
frame_skip_amount = 1
frame_skip = 0

# webcam id, 0 default camera, 2 is webcam
webcam_id = 0

# load the model weve trained
model_filename = f'model_{model_version}.{model_type}'
model_left_filename = f'model_{model_version}_left.{model_type}'
model_right_filename = f'model_{model_version}_right.{model_type}'
# model = tf.keras.models.load_model(model_filename, custom_objects={'custom_loss': custom_loss.custom_loss})
# model_left = tf.keras.models.load_model(model_left_filename, custom_objects={'custom_loss': custom_loss.custom_loss})
# model_right = tf.keras.models.load_model(model_right_filename, custom_objects={'custom_loss': custom_loss.custom_loss})

# load the model using tflite
main_interpreter = Interpreter(model_path=model_filename)
left_interpreter = Interpreter(model_path=model_left_filename)
right_interpreter = Interpreter(model_path=model_right_filename)

# allocate tensors
main_interpreter.allocate_tensors()
left_interpreter.allocate_tensors()
right_interpreter.allocate_tensors()

# get input and output tensors
main_input_details = main_interpreter.get_input_details()
left_input_details = left_interpreter.get_input_details()
right_input_details = right_interpreter.get_input_details()
main_output_details = main_interpreter.get_output_details()
left_output_details = left_interpreter.get_output_details()
right_output_details = right_interpreter.get_output_details()

# face and eye detector classifiers
face_cascade = cv.CascadeClassifier(f'{cv.data.haarcascades}haarcascade_frontalface_default.xml')
eye_cascade = cv.CascadeClassifier(f'{cv.data.haarcascades}haarcascade_eye.xml')

# number of recent predictions we take into account for picking side
# slows it down a lot
# might not be needed
num_predictions = 1


def predictEyeDirection(frame, box, side):
    (x, y, w, h) = box
    eye_frame = frame[y:y + h, x:x + w]
    eye_frame = cv.cvtColor(eye_frame, cv.COLOR_BGR2GRAY)
    # need to be same size as the models size
    if w < model_w or h < model_h:
        eye_frame = pad_image(eye_frame, model_w, model_h)
    elif w > model_w or h > model_h:
        eye_frame = crop_image(eye_frame, model_w, model_h)
    # eye_frame = crop_image(eye_frame, model_w, model_h)
    eye_frame = eye_frame.reshape((1, model_h, model_w, model_c))
    eye_frame = eye_frame.astype('float32')
    # predict
    if side == 'left':
        # predictions = model_left.predict(eye_frame)
        left_interpreter.set_tensor(left_input_details[0]['index'], eye_frame)
        left_interpreter.invoke()
        predictions = left_interpreter.get_tensor(left_output_details[0]['index'])
    elif side == 'right':
        # predictions = model_right.predict(eye_frame)
        right_interpreter.set_tensor(right_input_details[0]['index'], eye_frame)
        right_interpreter.invoke()
        predictions = right_interpreter.get_tensor(right_output_details[0]['index'])
    else:
        # predictions = model.predict(eye_frame)
        main_interpreter.set_tensor(main_input_details[0]['index'], eye_frame)
        main_interpreter.invoke()
        predictions = main_interpreter.get_tensor(main_output_details[0]['index'])
    class_index = np.argmax(predictions[0])
    if side == 'left':
        cv.putText(frame, f'{pred_to_name[class_index]}', (x + w, y - h), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255))
    elif side == 'right':
        cv.putText(frame, f'{pred_to_name[class_index]}', (x - w, y - h), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255))
    # return pred_to_name[class_index]
    return {f'{side} eye': pred_to_name[class_index]}


def getEyeRects(frame):
    left_eye = None
    right_eye = None
    # first find the face in the frame
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 3)
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        # box around face fine but eyes blocks the view
        cv.rectangle(frame, (x, y), (x + w, y + w), (150, 150, 0), 1)
        face_middle = (w / 2) + x
        face_middle_h = (h / 2) + y
        # now that we have the face we can find eyes only in that area
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 2, 0, minSize=(30, 30), maxSize=(60, 60))
        for (ex, ey, ew, eh) in eyes:
            cv.rectangle(frame, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (0, 150, 150), 1)
            if y + ey > face_middle_h:
                continue
            # check if its above the middle of face
            # cull amount is to remove eyebrow and under eye part
            cull_amount = round(eh * (1 / 4))
            # eye_middle = ((ex + ew) / 2) + x
            # eyes are mirrored with the webcam
            if x + ex < face_middle:
                right_eye = (x + ex, y + ey + cull_amount, ew, eh - (2 * cull_amount))
            else:
                left_eye = (x + ex, y + ey + cull_amount, ew, eh - (2 * cull_amount))
    return left_eye, right_eye


def crop_image(img, w, h):
    ih, iw = img.shape
    startx = int((iw - w) / 2)
    starty = int((ih - h) / 2)
    endx = startx + w
    endy = starty + h
    img = img[starty:endy, startx:endx]
    return img


def pad_image(img, w, h):
    ih, iw = img.shape
    padding = [(0, h - ih), (0, w - iw)]
    img = np.pad(img, padding, mode='constant', constant_values=0)
    return img

def track_eyes(fid):
    # Get the video frame
    ret, frame = cap.read()

    if not ret:
        print('Error reading from the camera, exiting')

    preds = []
    left_eye, right_eye = getEyeRects(frame)

    if not left_eye and not right_eye:
        print('Could not find an eye in current frame')

    # frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    if left_eye:
        pred = predictEyeDirection(frame, left_eye, 'left')
        preds.append(pred)
    if right_eye:
        pred = predictEyeDirection(frame, right_eye, 'right')
        preds.append(pred)

    # average of last couple predictions
    # for pred in preds:
    #     print(f'{pred}')
    
    cv.imwrite(f'frames/{fid}.png', frame)

    return preds


def close_capture():
    cap.release()


# start of the program
cap = cv.VideoCapture(webcam_id)
if not cap.isOpened():
    print('Cannot open camera, exiting')
    exit()

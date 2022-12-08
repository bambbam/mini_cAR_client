import tensorflow as tf
import numpy as np
import cv2
from time import sleep
import os
from dotenv import load_dotenv
from enum import Enum
import requests
import asyncio
load_dotenv()

if os.environ.get("mode")=="prod":
    from client.movement import Movement, handle_movement


classes =  [
    'Swiping Left',
    'Swiping Right',
    'Swiping Down',
    'Swiping Up',
    'Thumb Up',
    'Thumb Down',
    'Shaking Hand',
    'Stop Sign',
    'No gesture',
    'Doing other things'
]

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

class Conv3DModel(tf.keras.Model):
  def __init__(self):
    super(Conv3DModel, self).__init__()
    # Convolutions
    self.conv1 = tf.compat.v2.keras.layers.Conv3D(32, (3, 3, 3), name="conv1", data_format='channels_last')
    self.norm1 = tf.keras.layers.BatchNormalization()
    self.acti1 = tf.keras.layers.Activation('relu')
    self.pool1 = tf.keras.layers.MaxPool3D(pool_size=(2, 2, 2), data_format='channels_last')
    self.conv2 = tf.compat.v2.keras.layers.Conv3D(64, (3, 3, 3), name="conv1", data_format='channels_last')
    self.norm2 = tf.keras.layers.BatchNormalization()
    self.acti2 = tf.keras.layers.Activation('relu')
    self.pool2 = tf.keras.layers.MaxPool3D(pool_size=(2, 2,2), data_format='channels_last')

    # LSTM & Flatten
    self.convLSTM = tf.keras.layers.ConvLSTM2D(40, (3, 3))
    self.flatten =  tf.keras.layers.Flatten(name="flatten")

    # Dense layers
    self.d1 = tf.keras.layers.Dense(128, activation='relu', name="d1")
    self.out = tf.keras.layers.Dense(10, activation='softmax', name="output")

  def call(self, x):
    x = self.conv1(x)
    x = self.norm1(x)
    x = self.acti1(x)
    x = self.pool1(x)
    x = self.conv2(x)
    x = self.norm2(x)
    x = self.acti2(x)
    x = self.pool2(x)
    x = self.convLSTM(x)
    x = self.flatten(x)
    x = self.d1(x)

    return self.out(x)


class Prediction:
    def __init__(self, model):
        self.model = model
        self.to_predict = []
        self.num_frames = []
        self.cur_class = ''
    
    def predict(self, caps):
        assert len(caps) == 30
        
        to_predict = []
        for cap in caps[0::2]:
            gray = cv2.cvtColor(cap, cv2.COLOR_BGRA2GRAY)
            to_predict.append(cv2.resize(gray, (64,64)))
    
        frame_to_predict = np.array(to_predict, dtype=np.float32)
        frame_to_predict = frame_to_predict.reshape(-1, 15, 64, 64, 1)
        predict = self.model.predict(frame_to_predict)
        print(predict)
        classe = classes[np.argmax(predict)]
        self.cur_class = classe
        
        return self.cur_class

class CameraControl(Enum):
    getphoto = 0
    videostart = 1
    videostop = 2

async def request_server(ctrl : CameraControl, car_id: str):
    server_url = "http://" + os.environ.get("server_ip") + ':' + str(os.environ.get("server_port"))
    obj = {
        "car_id" : car_id,
        "ctrl" : ctrl.value,
    }
    requests.post(server_url+"/stream", json=obj)
    
async def handle_gesture(ges:str):
    print(ges)
    if os.environ.get("mode")=="dev":
        return
    if ges=="Swiping Left":        
        handle_movement(Movement.left.value)
    if ges=='Swiping Right':
        handle_movement(Movement.right.value)
    if ges=='Swiping Up':
        handle_movement(Movement.forward.value)
    if ges=='Swiping Down':
        handle_movement(Movement.backward.value)
    if ges=='Stop Sign':
        handle_movement(Movement.stop.value)
    async def asyncstop():
        await asyncio.sleep(3)
        handle_movement(Movement.stop.value)
    to_wait = []
    to_wait.append(asyncstop())
    
    if ges=='Thumb Up':
        to_wait.append(request_server(CameraControl.videostart, os.environ.get("car_id")))
    if ges=='Thumb Down':
        to_wait.append(request_server(CameraControl.videostop, os.environ.get("car_id")))
    if ges=='Shaking Hand':
        to_wait.append(request_server(CameraControl.getphoto, os.environ.get("car_id")))
    await asyncio.gather(*to_wait)
    
    

if __name__=="__main__":
    new_model = Conv3DModel()
    new_model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=tf.keras.optimizers.legacy.RMSprop())
    new_model.load_weights('weight/cp-0010.ckpt')
    pred = Prediction(new_model)
    cap = cv2.VideoCapture(0)
    classe =''
    
    while(True):
    # Capture frame-by-frame
        ret, frame = cap.read()

        # Our operations on the frame come here
        classe = pred.predict(frame)
            #sleep(0.1) # Time in seconds
            #font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, classe, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0),1,cv2.LINE_AA)


        # Display the resulting frame
        cv2.imshow('Hand Gesture Recognition',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

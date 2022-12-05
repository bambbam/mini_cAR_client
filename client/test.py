import tensorflow as tf
print(tf.__version__)
import numpy as np
import cv2

classes = [
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

# My model
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
    self.convLSTM =tf.keras.layers.ConvLSTM2D(40, (3, 3))
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

new_model = Conv3DModel()
new_model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=tf.keras.optimizers.legacy.RMSprop())
new_model.load_weights('weights/cp-0010.ckpt')

to_predict = []
num_frames = 0
cap = cv2.VideoCapture(0)
classe =''

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    to_predict.append(cv2.resize(gray, (64, 64)))
         
    if len(to_predict) == 30:
        frame_to_predict = np.array(to_predict, dtype=np.float32)[0::2]
        frame_to_predict = frame_to_predict.reshape(-1, 15, 64, 64, 1)
        predict = new_model.predict(frame_to_predict)   
        classe = classes[np.argmax(predict)]
        
        print('Classe = ', classe, 'Precision = ', np.amax(predict)*100,'%')

        to_predict = []
        
    cv2.putText(frame, classe, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0),1,cv2.LINE_AA)


    # Display the resulting frame
    cv2.imshow('Hand Gesture Recognition',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
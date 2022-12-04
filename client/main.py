from http import server
from typing import Any
import cv2
import numpy as np
import logging
import asyncio
from collections import deque
from itertools import islice

import asyncio_dgram

import pickle
import struct
import time
import sys
import math

# from client.movement import handle_movement
from multiprocessing import Process, Queue
from threading import Thread
from dotenv import load_dotenv
import os

from client.model import Conv3DModel, Prediction
import tensorflow as tf
from client.capture import Capture
from boto3 import client, resource


load_dotenv()
if os.environ.get("mode") == "prod":
    import RPi.GPIO as GPIO
    from client.movement import handle_movement

# import uuid
# car_id = uuid.uuid4().hex
# car_id = "e208d83305274b1daa97e4465cb57c8b"
car_id = os.environ.get("car_id")

# server_public_ip = "ec2-50-17-57-67.compute-1.amazonaws.com"

client_os = os.environ.get("client_os")
if client_os == "mac":
    MAX_DGRAM = 9216
else:
    MAX_DGRAM = 2**16
MAX_IMAGE_DGRAM = MAX_DGRAM - 64


frame_buffer = deque()

def frame_buffer_add(frame):
    global frame_buffer
    frame_buffer.append(frame)
    while len(frame_buffer) >= 60:
        frame_buffer.popleft()

def frame_buffer_get(num_frame):
    global frame_buffer
    return list(islice(frame_buffer, len(frame_buffer)-num_frame, len(frame_buffer)))


async def inference(server_ip):
    global frame_buffer
    new_model = Conv3DModel()
    new_model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=tf.keras.optimizers.legacy.RMSprop())
    new_model.load_weights('client/weight/cp-0004.ckpt')
    
    s3 = client('s3',
                aws_access_key_id = os.environ.get("aws_access_key_id"),
                aws_secret_access_key=os.environ.get("aws_secret_access_key"),
    )
    caputure = Capture(s3, os.environ.get("aws_bucket_name"), os.environ.get("car_id"))
    pred = Prediction(new_model)
    preded=''
    while True:
        if len(frame_buffer) >= 30:
            cur_preded = pred.predict(frame_buffer_get(30))
            if cur_preded!=preded:
                preded = cur_preded
                print(preded)
                if preded == 'Stop Sign':
                    caputure.upload_and_send_request(cv2.imencode('.png', frame_buffer_get(1)[0])[1].tobytes())
        time.sleep(1.0)
                
        
        
async def car_recieve(server_ip):
    try:
        reader, writer = await asyncio.open_connection(
            host=server_ip, port=os.environ.get("car_receive_port")
        )
    except:
        logging.warning("connection failed")
        return
    try:
        while True:
            buffer = b""
            recved_msg = "not received"
            while len(buffer) < 4:
                recved = await reader.read(4)
                if recved == 0:
                    return
                buffer += recved
                recved_msg = "received"
            movement = buffer[:4]
            buffer = buffer[4:]
            movement = struct.unpack("<L", movement)[0]
            if movement != 0:
                print(movement)
            if os.environ.get("mode") == "prod":
                handle_movement(movement)
            bin = pickle.dumps(recved_msg)
            writer.write(struct.pack("<L", len(bin)) + bin)
            await writer.drain()
    except KeyboardInterrupt:
        pass
    if os.environ.get("mode") == "prod":
        GPIO.cleanup()
    writer.close()


async def udpsending(server_ip):
    # car_id 먼저 보내고, 그 다음 jpgImg를 쪼개서 보낸다
    async def udp_send_car_id_and_jpg(car_id, jpgImg):

        car_idBin = car_id.encode("utf-8")
        jpgBin = pickle.dumps(jpgImg)

        jpgBin_size = len(jpgBin)
        num_of_fragments = math.ceil(jpgBin_size / (MAX_IMAGE_DGRAM))
        start_pos = 0

        await stream.send(car_idBin)  # car_id 전송

        while num_of_fragments:
            end_pos = min(jpgBin_size, start_pos + MAX_IMAGE_DGRAM)
            fragment = struct.pack("B", num_of_fragments) + jpgBin[start_pos:end_pos]
            # fragment 번호 + jpgImg fragment 전송
            # 맨 마지막 fragment 번호는 1
            await stream.send(fragment)
            start_pos = end_pos
            num_of_fragments -= 1

    stream = await asyncio_dgram.connect((server_ip, os.environ.get("frame_send_port")))
    VC = cv2.VideoCapture(0)

    print("\nClient Side")
    print("default = " + str(int(VC.get(cv2.CAP_PROP_FRAME_WIDTH))), end="x")
    print(str(int(VC.get(cv2.CAP_PROP_FRAME_HEIGHT))), end=" ")
    max_framerate = VC.get(cv2.CAP_PROP_FPS)
    print(str(int(max_framerate)) + "fps")

    width = int(os.environ.get("width"))
    height = int(os.environ.get("height"))
    framerate = int(os.environ.get("framerate"))
    jpeg_quality = int(os.environ.get("jpeg_quality"))

    VC.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    VC.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    fr_prev_time = 0

    print("current = " + str(int(VC.get(cv2.CAP_PROP_FRAME_WIDTH))), end="x")
    print(str(int(VC.get(cv2.CAP_PROP_FRAME_HEIGHT))), end=" ")
    print(
        str((lambda fr: max_framerate if fr > max_framerate else fr)(framerate)) + "fps"
    )

    while True:
        ret, cap = VC.read()
        fr_time_elapsed = time.time() - fr_prev_time
        if fr_time_elapsed > 1.0 / framerate:
            fr_prev_time = time.time()

            if os.environ.get("mode") == "prod":
                cap = cap[::-1]
            ret, jpgImg = cv2.imencode(
                ".jpg", cap, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
            )

            await udp_send_car_id_and_jpg(car_id, jpgImg)
            frame_buffer_add(cap)

def start_server(func_idx, server_ip):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    funcs = [udpsending, car_recieve, inference]
    asyncio.run(funcs[func_idx](server_ip))


def _asyncio():
    server_public_ip = os.environ.get("server_ip")
    if not server_public_ip:
        server_public_ip = "127.0.0.1"

    t1 = Thread(target=start_server, args=(0, server_public_ip))
    t1.start()
    t2 = Thread(target=start_server, args=(1, server_public_ip))
    t2.start()
    t3 = Thread(target=start_server, args=(2, server_public_ip))
    t3.start()
    
    
    t1.join()
    t2.join()
    t3.join()


if __name__ == "__main__":
    _asyncio()

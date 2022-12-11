from typing import Any
import cv2
import numpy as np
import logging
import asyncio
import uvloop
from collections import deque
from itertools import islice

import asyncio_dgram

import pickle
import struct
import time
import math

from threading import Thread, Lock

from client.model import Conv3DModel, Prediction
import tensorflow as tf

from client.model import handle_gesture

from client.movement import CarController, handle_movement
from client.config import get_settings

setting = get_settings()
if setting.mode.value == "prod":
    import RPi.GPIO as GPIO


# import uuid
# car_id = uuid.uuid4().hex
# car_id = "e208d83305274b1daa97e4465cb57c8b"
car_id = setting.car_id

# server_public_ip = "ec2-50-17-57-67.compute-1.amazonaws.com"

client_os = setting.client_os
if client_os == "mac":
    MAX_DGRAM = 9216
else:
    MAX_DGRAM = 2**16
MAX_IMAGE_DGRAM = MAX_DGRAM - 64


frame_buffer = deque()


lock = Lock()


def frame_buffer_add(frame):
    global frame_buffer, lock
    lock.acquire()
    frame_buffer.append(frame)
    while len(frame_buffer) >= 120:
        frame_buffer.popleft()
    lock.release()


def frame_buffer_get(num_frame):
    global frame_buffer, lock
    ret = []
    lock.acquire()
    for frame_idx in range(len(frame_buffer) - num_frame, len(frame_buffer)):
        ret.append(frame_buffer[frame_idx][::])
    lock.release()
    return ret


async def inference(server_ip):
    global frame_buffer
    new_model = Conv3DModel()
    new_model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=tf.keras.optimizers.legacy.RMSprop(),
    )
    new_model.load_weights("client/weight/myweight")
    pred = Prediction(new_model)
    preded = ""
    while True:
        lock.acquire()
        length = len(frame_buffer)
        lock.release()
        if length >= 60:
            cur_preded = pred.predict(frame_buffer_get(60))
            if cur_preded != preded:
                preded = cur_preded
                handle_gesture(preded)
        if setting.mode.value == "prod":
            handle_movement(9)
        time.sleep(2.0)
        if setting.mode.value == "prod":
            handle_movement(10)

async def car_recieve(server_ip):
    try:
        reader, writer = await asyncio.open_connection(
            host=server_ip, port=setting.car_receive_port
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
                CarController().set_control(movement)

            bin = pickle.dumps(recved_msg)
            writer.write(struct.pack("<L", len(bin)) + bin)
            await writer.drain()
    except KeyboardInterrupt:
        pass
    if setting.mode.value == "prod":
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

    stream = await asyncio_dgram.connect((server_ip, setting.frame_send_port))
    VC = cv2.VideoCapture(0)

    print("\nClient Side")
    print("default = " + str(int(VC.get(cv2.CAP_PROP_FRAME_WIDTH))), end="x")
    print(str(int(VC.get(cv2.CAP_PROP_FRAME_HEIGHT))), end=" ")
    max_framerate = VC.get(cv2.CAP_PROP_FPS)
    print(str(int(max_framerate)) + "fps)")

    width = int(setting.width)
    height = int(setting.height)
    framerate = int(setting.framerate)
    jpeg_quality = int(setting.jpeg_quality)

    VC.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    VC.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    VC.set(cv2.CAP_PROP_FPS, framerate)
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

            if setting.mode.value == "prod":
                cap = cap[::-1]
            ret, jpgImg = cv2.imencode(
                ".jpg", cap, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
            )

            await udp_send_car_id_and_jpg(car_id, jpgImg)
            frame_buffer_add(cap)


def start_server(func_idx, server_ip):
    loop = uvloop.new_event_loop()
    asyncio.set_event_loop(loop)
    funcs = [udpsending, car_recieve, inference]
    asyncio.run(funcs[func_idx](server_ip))


def run_car_control():
    CarController.run()


def _asyncio():
    server_public_ip = setting.server_ip
    if not server_public_ip:
        server_public_ip = "127.0.0.1"

    t1 = Thread(target=start_server, args=(0, server_public_ip))
    t1.start()
    t2 = Thread(target=start_server, args=(1, server_public_ip))
    t2.start()
    t3 = Thread(target=start_server, args=(2, server_public_ip))
    t3.start()
    t4 = Thread(target=run_car_control, args=())
    t4.start()

    t1.join()
    t2.join()
    t3.join()
    t4.join()


if __name__ == "__main__":
    _asyncio()

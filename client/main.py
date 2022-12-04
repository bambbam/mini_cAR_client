from http import server
from typing import Any
import cv2
import numpy as np
import logging
import asyncio
import asyncio_dgram
import pickle
import struct
import time
import sys
import math

# from client.movement import handle_movement
from multiprocessing import Process, Queue
from dotenv import load_dotenv
import os

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


def start_server(func_idx, server_ip):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    funcs = [udpsending, car_recieve]
    asyncio.run(funcs[func_idx](server_ip))


def _asyncio():
    server_public_ip = os.environ.get("server_ip")
    if not server_public_ip:
        server_public_ip = "127.0.0.1"

    t = Process(target=start_server, args=(0, server_public_ip))
    t.start()
    t = Process(target=start_server, args=(1, server_public_ip))
    t.start()


if __name__ == "__main__":
    _asyncio()


# async def sending(server_ip):

#     reader, writer = await asyncio.open_connection(
#         host=server_ip, port=os.environ.get("frame_send_port")
#     )

#     VC = cv2.VideoCapture(0)

#     print("\nClient Side")
#     print("default = " + str(int(VC.get(cv2.CAP_PROP_FRAME_WIDTH))), end="x")
#     print(str(int(VC.get(cv2.CAP_PROP_FRAME_HEIGHT))), end=" ")
#     max_framerate = VC.get(cv2.CAP_PROP_FPS)
#     print(str(int(max_framerate)) + "fps")

#     width = int(os.environ.get("width"))
#     height = int(os.environ.get("height"))
#     framerate = int(os.environ.get("framerate"))

#     VC.set(cv2.CAP_PROP_FRAME_WIDTH, width)
#     VC.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
#     fr_prev_time = 0

#     print("current = " + str(int(VC.get(cv2.CAP_PROP_FRAME_WIDTH))), end="x")
#     print(str(int(VC.get(cv2.CAP_PROP_FRAME_HEIGHT))), end=" ")
#     print(
#         str((lambda fr: max_framerate if fr > max_framerate else fr)(framerate)) + "fps"
#     )

#     while True:
#         ret, cap = VC.read()
#         fr_time_elapsed = time.time() - fr_prev_time
#         if fr_time_elapsed > 1.0 / framerate:
#             fr_prev_time = time.time()

#             # JPEG Quality [0,100], default=95
#             # 이미지에 따라 다르지만 대부분 70-80 이상부터 이미지 크기 급격히 증가
#             if os.environ.get("mode") == "prod":
#                 cap = cap[::-1]
#             ret, jpgImg = cv2.imencode(".jpg", cap, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
#             car_idBin = car_id.encode("utf-8")
#             jpgBin = pickle.dumps(jpgImg)

#             bin = car_idBin + jpgBin

#             writer.write(struct.pack("<L", len(bin)) + bin)
#             await writer.drain()

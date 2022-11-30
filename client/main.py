from http import server
from typing import Any
import cv2
import logging
import asyncio
import asyncio_dgram
import pickle
import struct
import time
# from client.movement import handle_movement
from multiprocessing import Process, Queue
from dotenv import load_dotenv
import os
load_dotenv()
if os.environ.get('mode')=='prod':
    import RPi.GPIO as GPIO
    from movement import handle_movement

# import uuid


# car_id = uuid.uuid4().hex
car_id = "e208d83305274b1daa97e4465cb57c8b"


server_public_ip = "ec2-50-17-57-67.compute-1.amazonaws.com"


async def car_recieve(server_ip):
    try:
        reader, writer = await asyncio.open_connection(host=server_ip, port=os.environ.get('car_receive_port'))
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
            if os.environ.get('mode')=='prod':
                handle_movement(movement)
            bin = pickle.dumps(recved_msg)
            writer.write(struct.pack("<L", len(bin)) + bin)
            await writer.drain()
    except KeyboardInterrupt:
        pass
    if os.environ.get('mode')=='prod':
        GPIO.cleanup()
    writer.close()


async def sending(server_ip):

    reader, writer = await asyncio.open_connection(host=server_ip, port=os.environ.get('frame_send_port'))

    VC = cv2.VideoCapture(0)

    print("\nClient Side")
    print("default = " + str(int(VC.get(cv2.CAP_PROP_FRAME_WIDTH))), end="x")
    print(str(int(VC.get(cv2.CAP_PROP_FRAME_HEIGHT))), end=" ")
    max_framerate = VC.get(cv2.CAP_PROP_FPS)
    print(str(int(max_framerate)) + "fps")


    width = int(os.environ.get('width'))
    height = int(os.environ.get('height'))
    framerate = int(os.environ.get('framerate'))


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

            # JPEG Quality [0,100], default=95
            # 이미지에 따라 다르지만 대부분 70-80 이상부터 이미지 크기 급격히 증가
            ret, jpgImg = cv2.imencode(".jpg", cap, [int(cv2.IMWRITE_JPEG_QUALITY), 50])

            car_idBin = car_id.encode("utf-8")
            jpgBin = pickle.dumps(jpgImg)

            bin = car_idBin + jpgBin

            writer.write(struct.pack("<L", len(bin)) + bin)
            await writer.drain()


async def udpsending(server_ip):

    stream = await asyncio_dgram.connect((server_ip, 9997))

    VC = cv2.VideoCapture(0)

    print("\nClient Side")
    print("default = " + str(int(VC.get(cv2.CAP_PROP_FRAME_WIDTH))), end="x")
    print(str(int(VC.get(cv2.CAP_PROP_FRAME_HEIGHT))), end=" ")
    max_framerate = VC.get(cv2.CAP_PROP_FPS)
    print(str(int(max_framerate)) + "fps")

    width = 320
    height = 180
    framerate = 30

    VC.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    VC.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    fr_prev_time = 0

    print("current = " + str(int(VC.get(cv2.CAP_PROP_FRAME_WIDTH))), end="x")
    print(str(int(VC.get(cv2.CAP_PROP_FRAME_HEIGHT))), end=" ")
    print(
        str((lambda fr: max_framerate if fr > max_framerate else fr)(framerate)) + "fps"
    )

    def fragment(arr, n):
        for i in range(0, len(datagram), n):
            yield arr[i : i + n]

    while True:
        ret, cap = VC.read()
        fr_time_elapsed = time.time() - fr_prev_time
        if fr_time_elapsed > 1.0 / framerate:
            fr_prev_time = time.time()
            ret, jpgImg = cv2.imencode(".jpg", cap, [int(cv2.IMWRITE_JPEG_QUALITY), 50])

            car_idBin = car_id.encode("utf-8")
            jpgBin = pickle.dumps(jpgImg)

            bin = car_idBin + jpgBin

            datagram = struct.pack("<L", len(bin)) + bin

            fragments = fragment(datagram, 1500)
            for fm in fragments:
                await stream.send(fm)


def start_server(func_idx, server_ip):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    funcs = [sending, udpsending, car_recieve]
    asyncio.run(funcs[func_idx](server_ip))


def _asyncio():
    server_public_ip = os.environ.get("server_ip")
    if not server_public_ip:
        server_public_ip = '127.0.0.1'

    t = Process(target=start_server, args=(0, server_public_ip))
    t.start()
    t = Process(target=start_server, args=(1, server_public_ip))
    t.start()
    t = Process(target=start_server, args=(2, server_public_ip))
    t.start()


if __name__ == "__main__":
    _asyncio()

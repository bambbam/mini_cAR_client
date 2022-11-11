from http import server
from typing import Any
import cv2
import logging
import asyncio
import pickle
import struct
from client.movement import handle_movement
from multiprocessing import Process, Queue
import RPi.GPIO as GPIO

# import uuid


# car_id = uuid.uuid4().hex
car_id = "e208d83305274b1daa97e4465cb57c8b"


server_public_ip = "ec2-50-17-57-67.compute-1.amazonaws.com"


async def car_recieve(server_ip):
    try:
        reader, writer = await asyncio.open_connection(host=server_ip, port=9998)
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
            handle_movement(movement)
            bin = pickle.dumps(recved_msg)
            writer.write(struct.pack("<L", len(bin)) + bin)
            await writer.drain()
    except KeyboardInterrupt:
        pass
    GPIO.cleanup()
    writer.close()


async def sending(server_ip):

    reader, writer = await asyncio.open_connection(host=server_ip, port=9999)

    VC = cv2.VideoCapture(0)
    while True:
        ret, cap = VC.read()
        ret, jpgImg = cv2.imencode(".jpg", cap)

        car_idBin = car_id.encode("utf-8")
        jpgBin = pickle.dumps(jpgImg)

        bin = car_idBin + jpgBin

        writer.write(struct.pack("<L", len(bin)) + bin)
        await writer.drain()


def start_server(func_idx, server_ip):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    funcs = [sending, car_recieve]
    asyncio.run(funcs[func_idx](server_ip))


def _asyncio():
    i = input("[1]127.0.0.1 [2]aws_ec2 [3]ip(public) (1/2/3)? ")
    if i == 1 or "":
        server_public_ip = "127.0.0.1"
    elif i == 2:
        server_public_ip = "ec2-50-17-57-67.compute-1.amazonaws.com"
    elif i == 3:
        server_public_ip = input("input server ip : ")
    t = Process(target=start_server, args=(0, server_public_ip))
    t.start()
    t = Process(target=start_server, args=(1, server_public_ip))
    t.start()


if __name__ == "__main__":
    _asyncio()

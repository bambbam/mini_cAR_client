from http import server
from typing import Any
import cv2
import logging
import asyncio
import pickle
import struct
import pydantic
from client.movement import handle_movement
from multiprocessing import Process, Queue
# import uuid


class Write(pydantic.BaseModel):
    car_id : str
    jpgImg : list

# car_id = uuid.uuid4().hex
car_id = "e208d83305274b1daa97e4465cb57c8b"


server_public_ip = "ec2-50-17-57-67.compute-1.amazonaws.com"

class Message(pydantic.BaseModel):
    message : str


async def car_recieve(server_ip):
    try:
        reader, writer = await asyncio.open_connection(
            host=server_ip, port=9998
        )
    except:
        logging.warning("connection failed")
        return        
    while True:
        buffer = b""
        recved_msg = "not received"
        while len(buffer) < 4:
            recved = await reader.read(4)
            if recved==0:
                return
            buffer += recved
            recved_msg = "received"
        movement = buffer[:4]
        buffer = buffer[4:]
        movement = struct.unpack("<L", movement)[0]
        if movement != 0:
            print(movement)
        handle_movement(movement)
        m = Message(message=recved_msg)
        bin = pickle.dumps(m.json())
        writer.write(struct.pack("<L", len(bin)) + bin)
        await writer.drain()
    writer.close()


async def sending(server_ip):
    
    reader, writer = await asyncio.open_connection(
        host=server_ip, port=9999
    )
    
    VC = cv2.VideoCapture(0)
    while True:
        ret, cap = VC.read()
        ret, jpgImg = cv2.imencode(".jpg", cap)
        to_write = Write(
            car_id = car_id,
            jpgImg = jpgImg.tolist()
        )
        bin = pickle.dumps(to_write.json())
        writer.write(struct.pack("<L", len(bin)) + bin)
        await writer.drain()



def start_server(func_idx, server_ip):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    funcs = [sending, car_recieve]
    asyncio.run(funcs[func_idx](server_ip))


def _asyncio():
    # server_public_ip = input("server ip: ")
    # if not server_public_ip:
    #     server_public_ip = "127.0.0.1"
    server_public_ip = "ec2-50-17-57-67.compute-1.amazonaws.com"
    t = Process(target=start_server, args=(0,server_public_ip))
    t.start()
    t = Process(target=start_server, args=(1,server_public_ip))
    t.start()


if __name__ == "__main__":
    _asyncio()

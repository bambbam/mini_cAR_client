from http import server
from typing import Any
import cv2
import logging
import asyncio
import pickle
import struct
import pydantic

# import uuid


class Write(pydantic.BaseModel):
    car_id : str
    jpgImg : list

# car_id = uuid.uuid4().hex
car_id = "e208d83305274b1daa97e4465cb57c8b"

server_public_ip = input("server ip: ")
if not server_public_ip:
    server_public_ip = "127.0.0.1"


def _asyncio():
    async def sending():
        try:
            reader, writer = await asyncio.open_connection(
                host=server_public_ip, port=9999
            )
        except:
            logging.warning("connection failed")
            return
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

    asyncio.run(sending())


if __name__ == "__main__":
    _asyncio()

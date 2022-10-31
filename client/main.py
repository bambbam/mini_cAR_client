from http import server
import cv2
import logging
import asyncio
import pickle
import struct

# import uuid

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

            car_idBin = car_id.encode("utf-8")
            jpgBin = pickle.dumps(jpgImg)

            bin = car_idBin + jpgBin

            writer.write(struct.pack("<L", len(bin)) + bin)
            await writer.drain()

    asyncio.run(sending())


if __name__ == "__main__":
    _asyncio()

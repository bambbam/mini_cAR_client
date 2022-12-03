from boto3 import client, resource
from uuid import uuid4
from datetime import datetime
import requests


def generate_now():
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y-%H-%M-%S")
    return dt_string

class Capture:
    def __init__(self, s3: client, bucket: str, car_id: str) -> None:
        self.s3 = s3
        self.bucket = bucket
        self.car_id = car_id
    
    def upload(self, data):
        try:
            key = generate_now()
            self.s3.put_object(Bucket= self.bucket, Key=self.car_id+'/'+key, Body=data)
            return key
        except Exception as e:
            print(e)
            
    def upload_video(self, video, prefix, key):
        raise NotImplementedError()
    
    
if __name__ == "__main__":
    # test code for capture
    import cv2
    from dotenv import load_dotenv
    import os
    load_dotenv()
    vc = cv2.VideoCapture(0)
    ret, cap = vc.read()
    
    server_url = os.environ.get("server_url") + ':' + str(os.environ.get("server_port"))
    
    
    s3 = client('s3',
                aws_access_key_id = os.environ.get("aws_access_key_id"),
                aws_secret_access_key=os.environ.get("aws_secret_access_key"),
    )
    videoCapture = Capture(s3, os.environ.get("aws_bucket_name"), os.environ.get("car_id"))
    key = videoCapture.upload(bytes(cap))
    obj = {
        "car_id" :os.environ.get("car_id"),
        "type" : "img",
        "key" : key
    }
    requests.post(server_url + "/gallery", obj)
    
from boto3 import client, resource
from uuid import uuid4
from datetime import datetime
import requests
import os
from dotenv import load_dotenv
load_dotenv()

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
            key = generate_now() + ".png"
            self.s3.put_object(Bucket= self.bucket, Key=self.car_id+'/'+key, Body=data)
            return key
        except Exception as e:
            print(e)
    
    def upload_and_send_request(self, data):
        server_url = "http://" + os.environ.get("server_ip") + ':' + str(os.environ.get("server_port"))
        key = self.upload(data)
        obj = {
            "car_id" :os.environ.get("car_id"),
            "type" : "img",
            "key" : key
        }
        print(server_url, obj)
        requests.post(server_url + "/gallery", json= obj)
        
    
    def upload_video(self, video, prefix, key):
        raise NotImplementedError()
    
    
if __name__ == "__main__":
    # test code for capture
    import cv2
    vc = cv2.VideoCapture(0)
    ret, cap = vc.read()
    for i in range(10):
        ret, cap = vc.read()
    s3 = client('s3',
                aws_access_key_id = os.environ.get("aws_access_key_id"),
                aws_secret_access_key=os.environ.get("aws_secret_access_key"),
    )
    videoCapture = Capture(s3, os.environ.get("aws_bucket_name"), os.environ.get("car_id"))
    key = videoCapture.upload_and_send_request(cv2.imencode('.png', cap)[1].tobytes())
    
import logging
import threading

import cv2
import imagezmq
import simplejpeg
import socket

from time import sleep

from werkzeug import Response, Request
from werkzeug.serving import run_simple

log = logging.getLogger(__name__)

SERVER_IP = 'localhost'


class ImageZMQStream:
    streams = {}
    quality = 95

    def run(self):
        try:
            while True:
                for stream, image in self.streams.items():
                    self.image_hub.send_reply(b'OK')

                    self.stream_monitor.send_image(stream, image)

        except Exception as e:
            log.debug("ImageZMQ Exception: {}".format(e))
            pass

    def __init__(self, IP_ADDRESS=SERVER_IP, HTTP_PORT=8090):
        log.info("ImageZMQ Stream starting at {}:{}".format(IP_ADDRESS, HTTP_PORT))
        self.image_hub = imagezmq.ImageHub()
        self.stream_monitor = imagezmq.ImageSender(connect_to='tcp://*:{}'.format(HTTP_PORT), REQ_REP=False)
        th = threading.Thread(target=self.run)
        th.daemon = True
        th.start()

    def send_frame(self, stream, frame):
        self.streams[stream] = frame
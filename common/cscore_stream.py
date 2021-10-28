#!/usr/bin/env python3

import cscore as cs
import cv2
import logging
import threading

from PIL import Image
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
from time import sleep

import simplejpeg

log = logging.getLogger(__name__)

SERVER_IP = 'localhost'

COLORSPACE = 'BGR'
QUALITY = 20


class CsCoreStream:
    def __init__(self, IP_ADDRESS=SERVER_IP, HTTP_PORT=8090, QUALITY=20, colorspace='BGR'):
        self.camera = cs.CvSource("cvsource", cs.VideoMode.PixelFormat.kMJPEG, 416, 416, 30)

        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 416)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 416)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        self.frame_to_send = []

        th = threading.Thread(target=self.output_frame, daemon=True)
        th.start()

        # start MJPEG HTTP Server
        log.info("CSCore Stream starting at {}:{}".format(IP_ADDRESS, HTTP_PORT))
        mjpegServer = cs.MjpegServer("httpserver", HTTP_PORT)
        mjpegServer.setSource(self.camera)

    def output_frame(self):
        while True:
            # retval, frame = self.cap.read(self.frame_to_send)
            # if retval:
            self.camera.putFrame(self.frame_to_send)

    def send_frame(self, frame):
        self.frame_to_send = frame



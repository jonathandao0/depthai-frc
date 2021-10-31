#!/usr/bin/env python3

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


# HTTPServer MJPEG
class VideoStreamHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        global QUALITY
        global COLORSPACE

        self.send_response(200)
        self.send_header('Content-type', 'multipart/x-mixed-replace; boundary=--jpgboundary')
        self.end_headers()
        while True:
            try:
                if hasattr(self.server, 'quality'):
                    QUALITY = self.server.quality
                    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), QUALITY]
                if hasattr(self.server, 'colorspace'):
                    COLORSPACE = self.server.colorspace

                if hasattr(self.server, 'frame_to_send'):
                    # image = Image.fromarray(cv2.cvtColor(self.server.frametosend, cv2.COLOR_BGR2RGB))
                    # stream_file = BytesIO()
                    # image.save(stream_file, 'JPEG')
                    self.wfile.write("--jpgboundary".encode())
                    frame = cv2.resize(self.server.frame_to_send, (320, 320))
                    # frame = self.server.frame_to_send
                    if COLORSPACE == 'BW':
                        img_str = cv2.imencode('.jpg', frame, encode_param)[1].tostring()
                    else:
                        img_str = simplejpeg.encode_jpeg(frame, quality=QUALITY, colorspace=COLORSPACE, fastdct=True)

                    self.send_header('Content-type', 'image/jpeg')
                    # self.send_header('Content-length', str(stream_file.getbuffer().nbytes))
                    self.send_header('Content-length', len(img_str))
                    self.end_headers()
                    # image.save(self.wfile, 'JPEG')

                    self.wfile.write(img_str)
                    self.wfile.write(b"\r\n--jpgboundary\r\n")
                    sleep(0.03)
            except Exception as e:
                log.error("MJPEG Exception: {}".format(e))
                self.flush_headers()
                self.finish()
                pass


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Handle requests in a separate thread."""
    pass


class MjpegStream:
    def __init__(self, IP_ADDRESS=SERVER_IP, HTTP_PORT=8090, QUALITY=20, colorspace='BGR'):
        # start MJPEG HTTP Server
        log.info("MJPEG Stream starting at {}:{}".format(IP_ADDRESS, HTTP_PORT))
        self.server_HTTP = ThreadedHTTPServer((IP_ADDRESS, HTTP_PORT), VideoStreamHandler)
        cfg = {
            'quality': QUALITY,
            'colorspace': colorspace
        }
        self.set_config(cfg)
        th = threading.Thread(target=self.server_HTTP.serve_forever)
        th.daemon = True
        th.start()

    def set_config(self, config):
        if 'quality' in config:
            self.server_HTTP.quality = config['quality']
        if 'colorspace' in config:
            self.server_HTTP.colorspace = config['colorspace']

    def send_frame(self, frame):
        self.server_HTTP.frame_to_send = frame



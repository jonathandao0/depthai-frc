#!/usr/bin/env python3

import cv2
import logging
import threading

from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
from time import sleep

import simplejpeg

log = logging.getLogger(__name__)

SERVER_IP = 'localhost'

QUALITY = 95


# HTTPServer MJPEG
class VideoStreamHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'multipart/x-mixed-replace; boundary=--jpgboundary')
        self.end_headers()
        while True:
            try:
                if hasattr(self.server, 'frame_to_send'):
                    # image = Image.fromarray(cv2.cvtColor(self.server.frametosend, cv2.COLOR_BGR2RGB))
                    # stream_file = BytesIO()
                    # image.save(stream_file, 'JPEG')
                    self.wfile.write("--jpgboundary".encode())
                    img_str = simplejpeg.encode_jpeg(self.server.frame_to_send, quality=QUALITY, colorspace='BGR', fastdct=True)
                    # img_str = cv2.imencode('.jpg', self.server.frame_to_send)[1].tostring()

                    self.send_header('Content-type', 'image/jpeg')
                    # self.send_header('Content-length', str(stream_file.getbuffer().nbytes))
                    self.send_header('Content-length', len(img_str))
                    self.end_headers()
                    # image.save(self.wfile, 'JPEG')

                    self.wfile.write(img_str)
                    self.wfile.write(b"\r\n--jpgboundary\r\n")
                    sleep(0.01)
            except Exception as e:
                log.debug("MJPEG Exception: {}".format(e))
                self.flush_headers()
                self.finish()
                pass


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Handle requests in a separate thread."""
    pass


class MjpegStream:
    def __init__(self, IP_ADDRESS=SERVER_IP, HTTP_PORT=8090):
        global QUALITY
        # start MJPEG HTTP Server
        log.info("MJPEG Stream starting at {}:{}".format(IP_ADDRESS, HTTP_PORT))
        self.server_HTTP = ThreadedHTTPServer((IP_ADDRESS, HTTP_PORT), VideoStreamHandler)
        th = threading.Thread(target=self.server_HTTP.serve_forever)
        th.daemon = True
        th.start()

    def send_frame(self, frame):
        self.server_HTTP.frame_to_send = frame



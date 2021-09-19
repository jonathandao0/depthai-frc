#!/usr/bin/env python3

import cv2
import threading

from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
from time import sleep

SERVER_IP = 'localhost'


# HTTPServer MJPEG
class VideoStreamHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'multipart/x-mixed-replace; boundary=--jpgboundary')
        self.end_headers()
        while True:
            try:
                if hasattr(self.server, 'frametosend'):
                    # image = Image.fromarray(cv2.cvtColor(self.server.frametosend, cv2.COLOR_BGR2RGB))
                    # stream_file = BytesIO()
                    # image.save(stream_file, 'JPEG')
                    self.wfile.write("--jpgboundary".encode())

                    img_str = cv2.imencode('.jpg', self.server.frametosend)[1].tostring()

                    self.send_header('Content-type', 'image/jpeg')
                    # self.send_header('Content-length', str(stream_file.getbuffer().nbytes))
                    self.send_header('Content-length', len(img_str))
                    self.end_headers()
                    # image.save(self.wfile, 'JPEG')

                    self.wfile.write(img_str)
                    self.wfile.write(b"\r\n--jpgboundary\r\n")
                    sleep(0.1)
            except Exception as e:
                print("MJPEG Exception: {}".format(e))
                pass


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Handle requests in a separate thread."""
    pass


class MjpegStream:
    def __init__(self, IP_ADDRESS=SERVER_IP, HTTP_PORT=8090):
        # start MJPEG HTTP Server
        self.server_HTTP = ThreadedHTTPServer((IP_ADDRESS, HTTP_PORT), VideoStreamHandler)
        th = threading.Thread(target=self.server_HTTP.serve_forever)
        th.daemon = True
        th.start()

    def sendFrame(self, frame):
        self.server_HTTP.frametosend = frame



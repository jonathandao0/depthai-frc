import logging

import cv2
import imagezmq

log = logging.getLogger(__name__)


def main():
    log.info("Starting Driverstation video stream client...")
    image_hub = imagezmq.ImageHub()

    stream_monitor = imagezmq.ImageSender(connect_to='tcp://*:5566', REQ_REP=False)

    while True:
        server, image = image_hub.recv_image()
        image_hub.send_reply(b'OK')
        stream_monitor.send_image(server, image)


if __name__ == '__main__':
    main()

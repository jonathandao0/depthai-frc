import time

import cv2


class FPSHandler:
    fps_color = (134, 164, 11)
    fps_type = cv2.FONT_HERSHEY_SIMPLEX

    def __init__(self, cap=None):
        self.timestamp = time.monotonic()
        self.start = None
        self.framerate = cap.get(cv2.CAP_PROP_FPS) if cap is not None else None
        self.useCamera = cap is None

        self.frame_cnt = 0
        self.ticks = {}
        self.ticks_cnt = {}

    def next_iter(self):
        if self.start is None:
            self.start = time.monotonic()

        if not self.useCamera:
            frame_delay = 1.0 / self.framerate
            delay = (self.timestamp + frame_delay) - time.monotonic()
            if delay > 0:
                time.sleep(delay)
        self.timestamp = time.monotonic()
        self.frame_cnt += 1

    def tick(self, name):
        if name in self.ticks:
            self.ticks_cnt[name] += 1
        else:
            self.ticks[name] = time.monotonic()
            self.ticks_cnt[name] = 0

    def tick_fps(self, name):
        if name in self.ticks:
            time_diff = time.monotonic() - self.ticks[name]
            return self.ticks_cnt[name] / time_diff if time_diff != 0 else 0
        else:
            return 0

    def fps(self):
        if self.start is None:
            return 0
        time_diff = self.timestamp - self.start
        return self.frame_cnt / time_diff if time_diff != 0 else 0

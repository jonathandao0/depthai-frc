#!/usr/bin/env python3

import os
import sys


if __name__ == '__main__':
    sys.path.append("/home/pi/depthai-frc")
    sys.path.append("/home/pi/depthai-frc/common")
    os.chdir("/home/pi/depthai-frc")
    os.system("python3 target-detection-host/main.py")

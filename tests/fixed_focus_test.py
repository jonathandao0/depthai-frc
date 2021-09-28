
import cv2
import depthai as dai

pipeline = dai.Pipeline()
pipeline.setOpenVINOVersion(dai.OpenVINO.Version.VERSION_2021_3)

camRgb = pipeline.createColorCamera()

camRgbControl = pipeline.createXLinkIn()
xoutRgb = pipeline.createXLinkOut()

camRgbControl.setStreamName("rgbControl")
xoutRgb.setStreamName("rgbFixed")

camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
camRgb.setInterleaved(False)
camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
camRgb.setFps(60)

camRgbControl.out.link(camRgb.inputControl)
camRgb.video.link(xoutRgb.input)

with dai.Device(pipeline) as device:
    outputQueue = device.getOutputQueue("rgbFixed", maxSize=4, blocking=False)
    controlQueue = device.getInputQueue("rgbControl")

    while True:
        cfg = dai.CameraControl()
        cfg.setManualFocus(150)
        controlQueue.send(cfg)
        frame = outputQueue.get().getCvFrame()

        cv2.imshow("Frame", frame)

        key = cv2.waitKey(1)

        if key == ord("q"):
            raise StopIteration()
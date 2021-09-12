import depthai

for device in depthai.Device.getAllAvailableDevices():
    print(f"{device.getMxId()} {device.state}")
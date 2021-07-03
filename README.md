# depthai-frc-localization
Implementing full-field localization using vision and depth measurements for FRC Robots. This specifically uses a deep learning model to identify key landmarks on the FRC field that have known locations. When the landmarks are identified, the measured distance to the landmarks from the camera are compared with what their ideal position should be. The difference is used to transform the outputted measurements to more accurately localize the robot's position on the field. This measurement is then used on the robot, in conjunction with its sensors to accurately determine its position on the field.


## Hardware
* [OAK-D](https://store.opencv.ai/products/oak-d)
* Raspberry Pi 4


## To Run
(Raspberry Pi Only) Follow The steps in the Luxonis documentation to install the base requirements:
https://docs.luxonis.com/projects/api/en/latest/install/#raspberry-pi-os

In the code directory, install the requirements:
`python3 -m pip install -r requirements.txt`

Run the program:
`python3 main.py`


## Performance
~26-27 FPS Running off a Raspbnerry Pi 4


## Future Improvements
* Feature Tracking: See [#146 Feature Tracking Stream Support](https://github.com/luxonis/depthai/issues/146). I assume that native feature tracking will improve the performance, so integrating this once it's release is a high priority.
* Proper triangulation for pose estimation: Currently, this makes a lot of dumb assuptions. Mostly to get this working as a proof of concept before trying to proceed further. Distance is calculated using 3D space and its not doing anything to project this to a flat 2D position, so the distances will be off. Either doing this or incorporating the heights of each target, as well as the camera will need to be done for more accurate measurements.


## Special Thanks
* Declan Freeman-Gleason (pietroglyph) of Team 4915: His [post](https://www.chiefdelphi.com/t/what-impressive-things-did-you-do-in-software-this-year/382245/48) on how he implemented full-field localization is pretty much what inspired me to look into this problem as being viable to implement.
* The WPILib Team: The massive improvements to the WPILib library over the last few years has pretty much enabled me to pursue problems like this due to making a lot of the complex math easy to implement.

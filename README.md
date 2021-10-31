# depthai-frc

A project I developed to implement real-time machine learning vision processing using the OopenCV AI Kit (OAK) devices. This was focused on using a model trained on objects for the 2020/2021 FRC Infinite Recharge game and using the model in several different applications.


## Demonstrations and Examples
Some videos showcasing this system can be found on my YouTube channel [here]().


## Paper
A paper detailing this project in more detail can be found [here]().


## Chief Dephi Thread
Discussion on this project can be found [here](https://www.chiefdelphi.com/).


## Dataset
Our dataset can be found on Supervisely [here](https://app.supervise.ly/share-links/k6PE2KAhDvpV8qtXCT1UYiY3MvhLvl7A5Vvkr4CXbIiZOhgt7Gv0q40l78Iqb2dG)


## Hardware
* [OAK-1](https://store.opencv.ai/products/oak-1)
* [OAK-D](https://store.opencv.ai/products/oak-d)
* Raspberry Pi 4 (2GB)


## Installation Steps
(Raspberry Pi Only) Follow The steps in the Luxonis documentation to install the base requirements:
https://docs.luxonis.com/projects/api/en/latest/install/#raspberry-pi-os

In the code directory, install the requirements:
`python3 -m pip install -r requirements.txt`

Each host is in its own python package, which can be run from the root directory:
`python3 [hostname]/main.py`


## Project Structure
The project is structure with various folders/packages. Here is a summary of what each folder/package does:

* **common**: Contains common functions and configs used for each host program
* **driverstaion-client**: A client program used to capture video streams from the host programs. Still a work-in-progress.
* **goal-depth-detection-host**: A standalone client used to detect the upper power ports for scoring. Uses an OAK-D device.
* **goal-depth-intake-detection-host**: A client used for both detecting the upper power ports for scoring and the powercells for intaking. Uses an OAK-D for goal detection and an OAK-1 for powercell detection.
* **intake-depth-detection-host**: A client used for both detecting powercells for intaking and for getting their positions relative to the robot. Uses an OAK-D device.
* **intake-detection-host**: A client used for both detecting powercells for intaking and to determine how many are in our indexing system. Uses two OAK-1 devices.
* **localization-host**: A client used to determine the robot's position by detecting fixed objects and using their detected position to determine the robot's position. Uses an OAK-D device.
* **object-detection-host**: A client used for both detecting the upper power ports for scoring and the powercells for intaking. Uses an OAK-1 for goal detection and an OAK-1 for powercell detection.
* **pipelines**: Contains code to generate pipelines for the OAK devices using the DepthAI libraries.
* **resources**: Contains reference images and the neural nework models in the form of .blob files for use with the OAK devices.
* **wpilib-startup-scripts**: Contains modified scripts to automatically run the programs on a WPILibPi image.


## Special Thanks
* **Declan Freeman-Gleason (pietroglyph)** of Team 4915: His [post](https://www.chiefdelphi.com/t/what-impressive-things-did-you-do-in-software-this-year/382245/48) on how he implemented full-field localization is what inspired me to look into implementing robot localization myself.
* **Yashas Ambati, Gabe Hart, Kevin Jaget, Marshall Massengill, and Team 900**: A lot of this work would not have been possible without the use of their open-source dataset. A link to their Zebravision 7.0 whitepaper can be found [here](https://team900.org/blog/ZebraVision-7.0/).
* **Austin and Team 971**: The localization implementation of using SIFT was heavily inspired by their summary of what they did for their 2020/2021 robot.
* **Peter Johnson, the WPILib Team, and the Robotpy Team**: The massive improvements to the WPILib library over the last few years has pretty much enabled me to pursue problems like this due to being able to leverage a lot of the framework they've developed over the years.
* **Luxonis**: The OAK devices and DepthAI libraries have been incredible at making this system possible from both cost and simplicity.

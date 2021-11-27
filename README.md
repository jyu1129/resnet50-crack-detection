# resnet50-crack-detection

<p align="center">
    Crack Detection is a tool that utilizes deep learning to recognize concrete wall cracks in pictures.
</p>
<p align="center">
<img src = "https://i.imgur.com/hRG0RHS.png" />
</p>

<p align="center">
<img src="https://i.imgur.com/2kf0vrI.png" />
<br />Cracks detected on a concrete wall
</p>

- [resnet50_crack_detection](#resnet50-crack-detection)
- [How to set up and run the project](#how-to-set-up-and-run-the-project)
  - [Prerequisites](#prerequisites)
  - [Installer](#installer)
  - [Web deployment](#web-deployment)
  - [Manual Install](#manual-install)
    - [Requirements](#requirements)
    - [Configuring JSON file](#configuring-json-file)
      - [mode](#mode)
      - [data](#data)
      - [training / validation / test](#training--validation--test)
      - [test_whole_image](#test_whole_image)
      - [model](#model)
    - [Running the project](#running-the-project)

# How to set up and run the project

-----
ResNet-50 Crack detection program is a Python program that will run on Windows Operating Systems but not tested on Linux, and macOS.

## Prerequisites
Machine learning entails a great deal of trial and error. The software is designed to experiment with thousands of  various parameters in order to find an algorithm that performs the result that are supposed to do. This program will take a very long time unless you have the necessary hardware.

Graphics cards, rather than CPUs, are better suited for the sort of computations that this program does. The training procedure must be executed on a PC with a GPU capable of doing so. When you do this on a CPU, it can take up to 16 hours to train your model, but it just takes a few hours on a GPU.

## Installer
There is an installer which installs everything for you and creates a desktop shortcut to launch straight into the GUI. You can find it from https://github.com/jyu1129/resnet50-crack-detection/releases.

## Web Deployment
If you are planning to just test it out, you can head over to the link provided. https://share.streamlit.io/jyu1129/resnet50-crack-detection/web/web.py

## Manual Install
Clone the repo with git or download the code from https://github.com/jyu1129/resnet50-crack-detection.

### Requirements
Run `pip install -r requirements.txt` to install all the dependencies that are required by this project to work.

### Configuring JSON file
There will be two main processes/modes that can be run here, which is training and predicting.

#### mode
Open the `config.json` file with any IDE or text editor. The mode is defaulted as `train` which configure the project to run on *train mode*. There will be three main processes/modes that can be run here, `train` which is *train mode* which train a new model based on the settings below in the `config.json` file; `predict-test` which is *predict on test dataset mode* which run predictions on test dataset with the model loaded; `predict` which is *predict on a single image mode* which run prediction on an image with the model loaded; `predict-live` which is *predict on a video streaming website mode* which run prediction on a live-streaming video with the URL loaded.

#### data
It is not recommended editing this field as it will break the process code in the project.

#### training / validation / test
This is to specify the filepath of the dataset for training process.

#### test_whole_image
This is to specify the filepath and the target size of the image for `predict` mode.

#### live
This is to specify the video streaming url and also the kernel size for `predict-live` mode.

#### model
It is also not recommended editing these fields except for `save_dir` and `load_dir` which is the save directory and load directory for the model(s) trained.

### Running the project
Run it with `python run.py`

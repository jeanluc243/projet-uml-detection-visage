import os

from roboflow import Roboflow

API_KEY = os.environ["ROBOFLOW_API_KEY"]
WORKSPACE = os.getenv("ROBOFLOW_WORKSPACE", "laurenes-workspace")
PROJECT = os.getenv("ROBOFLOW_PROJECT", "visagetrinome")
VERSION = int(os.getenv("ROBOFLOW_VERSION", "2"))

rf = Roboflow(api_key=API_KEY)
project = rf.workspace(WORKSPACE).project(PROJECT)
version = project.version(VERSION)

dataset = version.download("yolov8", location="./downloads")
print("Dataset téléchargé dans :", dataset.location)

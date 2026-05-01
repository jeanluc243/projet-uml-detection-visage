from roboflow import Roboflow

API_KEY = "J7io3ElnIcGinEQ2nfxP"
WORKSPACE = "laurenes-workspace"   # à vérifier exactement
PROJECT = "visagetrinome"
VERSION = 2

rf = Roboflow(api_key=API_KEY)
project = rf.workspace(WORKSPACE).project(PROJECT)
version = project.version(VERSION)

dataset = version.download("yolov8", location="./downloads")
print("Dataset téléchargé dans :", dataset.location)
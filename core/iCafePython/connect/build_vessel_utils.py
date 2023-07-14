import numpy as np
import warnings
from scipy.ndimage.interpolation import zoom
import torch
import math
import copy
import cv2
from skimage import measure
import pandas as pd

def data_preprocess(img):
    mean_intensity = np.mean(img)
    std_intensity = np.std(img)
    upper_bound = np.percentile(img, 99.5)
    lower_bound = np.percentile(img, 00.5)
    img = np.clip(img, lower_bound, upper_bound)
    # 防止除0
    img = (img - mean_intensity) / (std_intensity + 1e-9)
    img = np.array([img])
    img = torch.from_numpy(img)
    return img.unsqueeze(0)


def get_shell(fl_Num_Points, fl_Radius):
    x_list = []
    y_list = []
    z_list = []
    offset = 2.0 / fl_Num_Points
    increment = math.pi * (3.0 - math.sqrt(5.0))

    for i in range(fl_Num_Points):
        z = ((i * offset) - 1.0) + (offset / 2.0)
        r = math.sqrt(1.0 - pow(z, 2.0))

        phi = ((i + 1) % fl_Num_Points) * increment

        x = math.cos(phi) * r
        y = math.sin(phi) * r
        x_list.append(fl_Radius * x)
        y_list.append(fl_Radius * y)
        z_list.append(fl_Radius * z)
    return x_list, y_list, z_list


def prob_terminates(pre_y, max_points):

    res = torch.sum(-pre_y * torch.log2(pre_y))
    return res / torch.log2(torch.from_numpy(np.array([max_points])).float())

def get_shell(fl_Num_Points, fl_Radius):
    x_list = []
    y_list = []
    z_list = []
    offset = 2.0 / fl_Num_Points
    increment = math.pi * (3.0 - math.sqrt(5.0))

    for i in range(fl_Num_Points):
        z = ((i * offset) - 1.0) + (offset / 2.0)
        r = math.sqrt(1.0 - pow(z, 2.0))

        phi = ((i + 1) % fl_Num_Points) * increment

        x = math.cos(phi) * r
        y = math.sin(phi) * r
        x_list.append(fl_Radius * x)
        y_list.append(fl_Radius * y)
        z_list.append(fl_Radius * z)
    return x_list, y_list, z_list


def get_angle(v1, v2):
    cosangle = v1.dot(v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    cosangle = np.clip(cosangle, -1, 1)
    return math.degrees(np.arccos(cosangle))

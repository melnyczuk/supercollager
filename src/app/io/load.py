# 'faster_rcnn_resnet50_v1b_voc'
from gluoncv import model_zoo  # type: ignore


def gluoncv_model(model_name: str, pretrained=True):
    return model_zoo.get_model(model_name, pretrained=pretrained)

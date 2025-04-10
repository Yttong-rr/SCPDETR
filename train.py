import warnings
warnings.filterwarnings('ignore')
from ultralytics import RTDETR

if __name__ == '__main__':
    model = RTDETR('ultralytics/cfg/models/rt-detr/rtdetr-SCP.yaml')
    #model.load('D:\\BaiduNetdiskDownload\\weights\\rtdetr-r50.pt') # loading pretrain weights
    model.train(data='pcbdata.yaml',
                cache=False,
                imgsz=640,
                epochs=300,
                batch=8,
                workers=4,
                device='0',
                # resume='', # last.pt path
                project='runs/train',
                name='exp',
                )
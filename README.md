# Object Detection MobileNet SSD
Deep Learning 모델을 이용하여을 객체를 검출하는 방법에 대해 알아보도록 하겠습니다.

Object Detection 분야에 딥러닝을 최초로 적용시킨 모델이 2013년 11월에 등장하는데 그 모델이 바로 R-CNN(Regions with Convolutional Neuron Networks features) 입니다. 분명 기존의 다른 모델과 비교해 성능을 상당히 향상시킨 모델이였지만 처리속도가 매우 느려서 Real-Time에서 활용하기 어렵습니다. (실제로 이미지 한장단 GPU환경에서는 13초가 걸렸으며 CPU로는 53초가 걸렸습니다.) 이후 수많은 Deep Learning 이용한 모델들이 등장하기 시작하는데, 그들의 고민 중 하나가 바로 처리속도 였습니다. 

#### MobileNet SSD

MobileNet은 모바일 및 임베디드 비전 애플리케이션용으로 설계된 경량 심층 신경망 아키텍처입니다. 현실에서 사용되는 실제 응용 프로그램은 제한된 장치(Raspberry Pi, 스마트폰 등)에서 적시에 이루어질 수 있어야 하는데 이러한 요구 사항을 충족하기 위해 개발된 것이 MobileNet(2017년)입니다.

<div align="center">
  <img src="https://blog.kakaocdn.net/dn/dvWPj8/btrSqX4pVFe/7vwsokr27MrZN0aUdOrT8K/img.png" width="50%">
</div>

거의 같은 시기에(2016년) Google 연구 팀도 정확도를 크게 떨어뜨리지 않고 임베디드 장치에서 실시간으로 실행할 수 있는 모델에 대한 요구를 충족하기 위해 SSD(Single Shot Detector)를 개발합니다.

<div align="center">
  <img src="https://blog.kakaocdn.net/dn/mq3YI/btrSnVsVEwo/378448mL8ko95il6lyeXU0/img.png" width="50%">
</div>

두 연구 모두 저사양 장치에서 리소스가 많고 전력 소모가 많은 신경망을 실행하는 어려움 해결하고자하는 공통적인 목표가 있었고 결국 MobileNet이 SSD에서 기본 네트워크로 사용되면서 **MobileNet SSD** 이 됩니다. (SSD는 기본 네트워크와 독립적으로 설계되었고 VGG, YOLO 또는 MobileNet과 같은 기본 네트워크 위에서 실행될 수 있었습니다.) 

------

물론 이후에 훌륭한 모델이 많이 나왔습니다. 하지만 이 글에서는 간단하게 Deep Learnig 모델을 활용하여 객체를 검출하는 방법을 소개하기 위함이니 MobileNet SSD를 사용하도록 하겠습니다. 실제로 OpenCV를 사용하여 매우 간단하게 사용 할 수 있습니다.

사전 학습된 [MobileNet SSD 모델](https://drive.google.com/file/d/10yMi2mvkZpHSwtMkfS6-InhI0cc9alLT/view?usp=sharing) 및 [prototxt](https://drive.google.com/file/d/1EP4fbqOah6aQ2HMPeQSw3ezzAL2VEdNh/view?usp=sharing)를 다운로드합니다.

#### **Import packages**

```python
import numpy as np
import cv2
import matplotlib.pyplot as plt
```

#### **Function declaration**

Jupyter Notebook 및 Google Colab에서 이미지를 표시할 수 있도록 Function으로 정의

```python
def img_show(title='image', img=None, figsize=(8 ,5)):
    plt.figure(figsize=figsize)
 
    if type(img) == list:
        if type(title) == list:
            titles = title
        else:
            titles = []
 
            for i in range(len(img)):
                titles.append(title)
 
        for i in range(len(img)):
            if len(img[i].shape) <= 2:
                rgbImg = cv2.cvtColor(img[i], cv2.COLOR_GRAY2RGB)
            else:
                rgbImg = cv2.cvtColor(img[i], cv2.COLOR_BGR2RGB)
 
            plt.subplot(1, len(img), i + 1), plt.imshow(rgbImg)
            plt.title(titles[i])
            plt.xticks([]), plt.yticks([])
 
        plt.show()
    else:
        if len(img.shape) < 3:
            rgbImg = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            rgbImg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
 
        plt.imshow(rgbImg)
        plt.title(title)
        plt.xticks([]), plt.yticks([])
        plt.show()
```

#### **Load Model**

위에서 다운로드 한 모델과 prototxt 파일을 model이라는 폴더를 만들어 그 하위에 복사하였습니다.

```python
prototxt_path = 'model/MobileNetSSD_deploy.prototxt.txt'
model_path = 'model/MobileNetSSD_deploy.caffemodel'
 
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
```

MobileNet SSD Model에 학습된 Class를 List로 정의합니다.

```python
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", 
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]
 
LABEL_COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
```

#### **Load Image**

테스트 할 이미지를 Load합니다.

```python
cv2_image = cv2.imread('asset/images/topgun.jpg', cv2.IMREAD_COLOR)
img_show('original image', cv2_image)
```

<div align="center">
  <img src="https://blog.kakaocdn.net/dn/l9cOe/btrSsmW15O4/rgWqaNaz7C9t0uKwkkW5N0/img.png" width="50%">
</div>

#### **Object Detection**

이미지 크기에 따라서 다를 수는 있지만 대체적으로 다른 Model에 비해 상대적으로 빠릅니다. 

```python
(h, w) = cv2_image.shape[:2]
resized = cv2.resize(cv2_image, (300, 300))
blob = cv2.dnn.blobFromImage(resized, 0.007843, (300, 300), 127.5)
 
net.setInput(blob)
detections = net.forward()
```

검출된 객체의 confidence 값이 특정 임계치 이상 일 경우만 표시 하도록 합니다. 임계치는 conf로 선언했고 저는 20% 이상인 대상만 추출하도록 했습니다. 

```python
conf = 0.2
vis = cv2_image.copy()
 
# 추출된 영역을 반복 수행 confidence 값이 임계치를 넘는 경우만 표시
for i in np.arange(0, detections.shape[2]):
    confidence = detections[0, 0, i, 2]
 
    if confidence > conf:
        idx = int(detections[0, 0, i, 1])
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
        
        print("[INFO] {} : [ {:.2f} % ]".format(CLASSES[idx], confidence * 100))
        
        cv2.rectangle(vis, (startX, startY), (endX, endY), LABEL_COLORS[idx], 1)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.putText(vis, "{} : {:.2f}%".format(CLASSES[idx], confidence * 100), (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, LABEL_COLORS[idx], 2)
```

Output:

```shell
[INFO] person : [ 99.29 % ]
[INFO] motorbike : [ 97.91 % ]
[INFO] aeroplane : [ 26.47 % ]
```

Object를 표시한 이미지를 확인합니다.

```python
img_show('Object Detection', vis, figsize=(16,10))
```

<div align="center">
  <img src="https://blog.kakaocdn.net/dn/bQg2Bf/btrSqNOAKIF/hkv8UTKUKpCKaFfPzwMzO0/img.png" width="50%">
</div>

------

결과는 매우 우수한 편입니다. 몇가지 더 테스트를 해보았는데 대체적으로 괜찮은 것 같습니다.

<div align="center">
  <img src="https://blog.kakaocdn.net/dn/IZhbw/btrSs1kNenY/5VsLkpkS4CSRXBuHFXRDM0/img.png" width="50%">
  <img src="https://blog.kakaocdn.net/dn/crkd1x/btrSqN17lKF/knWwoXM7XKJKAU2gRsjBJ0/img.png" width="50%">
</div>

테스트 데이타를 찾다가 존윅 이라는 영화의 한 장면이 생각나서 찾아 테스트 했는데 존윅에 출연한 개를 자꾸 '소'라고 인식하네요.

<div align="center">
  <img src="https://blog.kakaocdn.net/dn/b7kPbO/btrSqj1nJ3k/XetS9poVqYMGM2ptMUVnH0/img.png" width="50%">
</div>

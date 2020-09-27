# YOLO train

- 기본적으로 <a href=https://github.com/qqwweee/keras-yolo3>keras-yolo3</a>를 참고하세요.   
  (저는 달라진 tensorflow, keras 문법 수정 및 데이터셋을 바꿨고 사용하지 않는 코드는 삭제했습니다.)
- Darknet, GPU 기반으로 동작하므로 본인의 컴퓨터 환경에 따라 실행이 안 될 수도 있습니다.     
- 기존 YOLO 대신 더 가벼운 tiny YOLO를 사용합니다.    

<br/>

## Installation 
- **pip install numpy**
- **pip install tensorflow**
  (만약 현재 tensorflow 버전이 2.x가 아니라 1.x라면 pip uninstall tensorflow 실행 후 다시 명령어 실행)
- **pip install keras==2.2.4**
- **pip install image**
- 프로그램이 실행되지 않을 시, <a href=https://github.com/qqwweee/keras-yolo3>keras-yolo3</a>의 test 환경을 참고하세요.

<br/>

## Dataset
- example 폴더 하위의 dataset 폴더에 저장되어 있습니다. (출처: https://www.kaggle.com/tannergi/microcontroller-detection)
- 4개의 클래스 (Arduino_Nano, Heltec_ESP32_Lora, ESP8266, Raspberry_Pi_3) 로 구성되어 있으며 600X800의 이미지입니다.    
- 데이터는 이미 <a href=https://github.com/qqwweee/keras-yolo3>keras-yolo3</a>의 학습 환경에 맞게 convert 해둔 상태입니다.    
  직접 convert 하고 싶으시다면 dataset 폴더의 압축파일을 풀고, 해당 데이터에 대해 <code>mv.py</code> 실행 후 <code>voc_annotation.py</code> 를 실행시켜주시면 됩니다.

<br/>

## Customizing
<code>train.py </code>의 main함수를 수정합니다.    
- **annotation_path** = '../dataset/train_all.txt'** # Convert Annotation의 결과 파일경로
- **log_dir** = 'logs/000/' # 학습 결과 저장 경로    
- **classes_path** = 'model_data/class.txt' # 학습하는 클래스 목록    
- **anchors_path** = 'model_data/tiny_yolo_anchors.txt' # tiny YOLO 모델의 anchors    
- **weights_path** ='model_data/yolo_weights.h5' # Convert Darknet Model To Keras Model 결과 h5파일 (이미 convert 처리되어 있음)

<br/>

## 주의사항
- 컬러 이미지이며 이미지 크기가 크기 때문에 학습량이 많으므로 RAM 부족으로 인해 실행이 안 될 수 있습니다.    
  (이미지의 크기를 줄여서 학습시키는 방법이 존재합니다.)
- 라이브러리의 버전 문제로 인해 실행 중 오류가 날 가능성이 높습니다.


<br/>

## Reference
- https://github.com/qqwweee/keras-yolo3
- https://nero.devstory.co.kr/post/pj-too-real-03/
- https://gilberttanner.com/blog/yolo-object-detection-with-keras-yolo3

# Visual Recognition
YOLO를 이용한 **Object detection** 

## Object detection architecture 
![image](https://user-images.githubusercontent.com/46422952/133761193-d4bbfb5d-d376-4fbe-bb42-3e21da7c51c5.png) [1]

    1. Backbone: input을 feature map을 추출
    2. Head: 추출한 feature map을 통해 classification 및 localization 수행
        (1) Dense prediction (One-stage detector)
            - Classification과 bounding box regression 부분이 통합
        (2) Sparse prediction (Two-stage detector)
            - Dense와 달리 두 작업을 분리
    3. Neck: Backbone과 Head의 연결 및 feature map의 정제와 재구성

## You Only Look Once (YOLO)
![image](https://user-images.githubusercontent.com/46422952/133784464-1e3fde6c-f5a5-4817-857b-f754a9a69327.png) [2]    

    : CNN 기반의 object detection algorithm
    : 다양한 버전이 공개되었으나 실습에서는 v4를 사용
    : YOLOv4 architecture
        - Backbone: CSP-Darknet53 
        - Neck: SPP, PANet
        - Head: YOLOv3

## Library for YOLO
    1. OpenCV
        : Real-time image processing을 위한 computer vision library    
        : Windows, Linux 지원
        : 공식적으로 CPU만 지원 (GPU를 사용할 경우, CUDA 등을 활용하여 직접 compile)
        : C, C++ 기반이나 python으로 binding하여 사용 가능

    2. Darknet    
        : YOLO 개발자가 만든 open-source deep learning framework
        : 기본적으로 Linux만 지원 (Windows에서 사용할 경우, 추가적인 작업 필요)
        : CPU, GPU 지원
        : C, CUDA 기반
        : 다양한 image processing을 위해 OpenCV 활용 가능

<br>

***
<br>

## GOAL
    1. custom data를 통해 YOLOv4 model 학습
    2. 학습된 model을 통해 object detection 수행 

## Environment    
    - OS: Ubuntu 18.04 
    - GPU: GeForce RTX 3080    
    - CUDA 11.4, CuDNN 8.2.1
    - Language: C, Python 3.7.11 (Anaconda3)
    
## Install    
0. Darknet을 위한 OpenCV 설치 [3]    
    : version 3.4.0   
    : Darknet compile 시 사용하지 않을 경우 설치 불필요  

    ```
    // Ubuntu의 기본 OpenCV 삭제
    sudo apt-get remove libopencv* 
    
    // 개발자 도구 설치
    sudo apt-get update
    sudo apt-get upgrade
    sudo apt-get install build-essential cmake unzip pkg-config

    // library 설치
    sudo apt-get install libjpeg-dev libpng-dev libtiff-dev
    sudo apt-get install libavcodec-dev libavformat-dev libswscale-dev libv4l-dev v4l-utils libxvidcore-dev libx264-dev libxine2-dev
    sudo apt-get install libgtk-3-dev
    sudo apt-get install mesa-utils libgl1-mesa-dri libgtkgl2.0-dev libgtkglext1-dev
    sudo apt-get install libatlas-base-dev gfortran libeigen3-dev
    sudo apt-get install python2.7-dev python3-dev python-numpy python3-numpy // 개인적으로 이건 설치하지 않아도 될 듯

    // OpenCV download
    mkdir opencv 
    cd opencv 
    wget -O opencv.zip https://github.com/opencv/opencv/archive/3.4.0.zip 
    wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/3.4.0.zip

    unzip opencv.zip 
    unzip opencv_contrib.zip

    // build    
    cd opencv-3.4.0 
    mkdir build 
    cd build

    cmake -D CMAKE_BUILD_TYPE=RELEASE \ -D BUILD_opencv_cudacodec=OFF \ -D CMAKE_INSTALL_PREFIX=/usr/local \ -D WITH_TBB=OFF \ -D WITH_IPP=OFF \ -D WITH_1394=OFF \ -D BUILD_WITH_DEBUG_INFO=OFF \ -D BUILD_DOCS=OFF \ -D INSTALL_C_EXAMPLES=ON \ -D INSTALL_PYTHON_EXAMPLES=ON \ -D BUILD_EXAMPLES=OFF \ -D BUILD_TESTS=OFF \ -D BUILD_PERF_TESTS=OFF \ -D WITH_QT=OFF \ -D WITH_GTK=ON \ -D WITH_OPENGL=ON \ -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib-3.4.0/modules \ -D WITH_V4L=ON \ -D WITH_FFMPEG=ON \ -D WITH_XINE=ON \ -D BUILD_NEW_PYTHON_SUPPORT=ON \ -D PYTHON2_INCLUDE_DIR=/usr/include/python2.7 \ -D PYTHON2_NUMPY_INCLUDE_DIRS=/usr/lib/python2.7/dist-packages/numpy/core/include/ \ -D PYTHON2_PACKAGES_PATH=/usr/lib/python2.7/dist-packages \ -D PYTHON2_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython2.7.so \ -D PYTHON3_INCLUDE_DIR=/usr/include/python3.6m \ -D PYTHON3_NUMPY_INCLUDE_DIRS=/usr/lib/python3/dist-packages/numpy/core/include/ \ -D PYTHON3_PACKAGES_PATH=/usr/lib/python3/dist-packages \ -D PYTHON3_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.6m.so \ ../

    // compile
    make -j20 // 숫자는 할당하고 싶은 core 개수만큼

    sudo make install 
    sudo sh -c echo '/usr/local/lib/' > sudo /etc/ld.so.conf.d/opencv.conf 
    sudo ldconfig
    ```
    <br>

1. Darknet 설치    
    1) ``` git clone https://github.com/AlexeyAB/darknet ```    
    <br>
    2) Makefile 상황에 맞게 수정       
    
    - GPU, CuDNN, OpenCV 등 사용할 option 설정       
            ![image](https://user-images.githubusercontent.com/46422952/133800076-8e2dc630-102f-44e4-be99-38001a4f74b8.png)    

    - 본인의 CUDA 설치 폴더 확인 및 변수(COMMON, LDFLAG) 수정    
            ![image](https://user-images.githubusercontent.com/46422952/133802407-68d9e977-afe1-4249-9aaa-fc7d05bd880a.png)    
    
    - NVCC 변수 수정    
            ![image](https://user-images.githubusercontent.com/46422952/133803851-5fbfd47a-3d0c-4030-b27e-d7a1fe55bbbd.png)    
            
    <br>

    3) Build  
    ``` make ```  
    <br> 
    
    4) Test    
    (1) coco dataset에 학습된 yolov4 weight 다운로드    
    ``` wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights ```    
    (2) 실행 (example) 
        - Image    
            ```sh image_yolo4.sh```    
        - Video   
            ```sh video_yolo4.sh```   
        - Web cam     
        ```sh net_cam_v4.sh```

<br>

2. Python module 설치    
    ``` pip install numpy==1.20.3 ```     
    ``` pip install opencv-python==4.5.3.56 ```


<br>

***
<br>

## Dataset
1. Train, test image
     
    ![image](https://user-images.githubusercontent.com/46422952/134033493-a88d830d-2e5a-4eeb-8451-3044506c93a7.png)     
    : train 10장, test 2장 및 video      

<br>

2. Labeling
    1. Yolo format     
        ![image](https://user-images.githubusercontent.com/46422952/134041409-58f3209d-56bc-4c42-b80f-f78af1fed401.png)    
        : box의 크기 값은 전체 크기에 대한 비율 값    
        : 각 image마다 annotation이 적힌 한 개의 txt file 존재
        - class number
        - center x
        - center y
        - w
        - h     
        <br>
    2. Tool (Yolo-mark)    
        (1) Download    
        ``` git clone https://github.com/AlexeyAB/Yolo_mark ```

        (2) Set data     
        - <u>Yolo_mark/x64/Release/data/img</u>에 있는 모든 file 삭제
        - 동일 directory에 자신의 data 복사
        - <u>Yolo_mark/x64/Release/data/obj.data</u>의 classes 수정
        - <u>Yolo_mark/x64/Release/data/obj.name</u>의 class name 수정 

            <img width="515" alt="image" src="https://user-images.githubusercontent.com/46422952/134046002-1b299bd2-9a14-4a6d-a443-fe1779e42199.png">      

        (3) Run
        <u>Yolo_mark</u>에 있는 linux_mark.sh 실행 ( ``` sh linux_mark.sh ``` )
            <img width="1147" alt="image" src="https://user-images.githubusercontent.com/46422952/134043442-446b1de6-eda2-4bd0-babe-69087a7f31ad.png">
        프로그램 종료는 esc     

<br>

## Training
1. Setting [4,6]    
    1) Edit cfg file     
        : <u>darknet/cfg</u>에 있는 yolov4.cfg 편집

        <img width="200" alt="image" src="https://user-images.githubusercontent.com/46422952/134051666-75f478b0-20ea-48ec-9e5c-371488f6d700.png"> 

        - subdivision = 16 (학습 문제때문에 64로 변경)            
        - width, height = 416     
        - max_batches = class 수 * 2000      
        - steps = max_batches의 80, 90%              
        <br>

        : 3개의 [yolo] 부분과 그 전의 [convolutional] 수

        <img width="198" alt="image" src="https://user-images.githubusercontent.com/46422952/134051873-741e4d91-5cdc-4325-995d-5fece6201375.png">      
        
        - filters = (class 수 + 5) * 3
        - classes = class 수     
        <br>
    2) Set data    
        : <u>Yolo_mark/x64/Release/data/img</u> directory를 <u>darknet/data</u> 밑으로 복사     
        : <u>Yolo_mark/x64/Release/data/</u>에 있는 obj.data, obj.names, train.txt를 <u>darknet/data</u> 밑으로 복사    
        : train.txt의 경로가 image 경로와 동일한 지 확인 및 수정
        <img width="793" alt="image" src="https://user-images.githubusercontent.com/46422952/134056589-45babaf4-6162-4760-a66e-ae299879d799.png">    
        <br>
    3) Download weight file    
        : darknet에서 제공하는 pre-trained model 사용    
        ``` wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.conv.137 ```      
        <br>
2. Train    
    ``` ./darknet detector train data/obj.data cfg/yolov4.cfg yolov4.conv.137 -map ```    
    
    ![chart_yolov4](https://user-images.githubusercontent.com/46422952/134046849-f02a3f6f-dd35-40b3-9f24-f2101a24647f.png)    
    <br>

3. Save weight    
    : <u> darknet/backup</u> directory에 weight가 iteration 1000마다 자동으로 저장 

    <img width="463" alt="image" src="https://user-images.githubusercontent.com/46422952/134060988-edbc8a5f-26e6-4ade-84f6-b6e610cd3dec.png">       
    
<br>

## Detection
1. Image    
    ``` ./darknet detector test ./data/obj.data ./cfg/yolov4.cfg ./backup/yolov4_best.weights <test image 경로> --ext_output```   

    ![predictions](https://user-images.githubusercontent.com/46422952/134036292-12fe0b45-5cc7-4a00-8b09-ebf319181de6.jpg)      

    또는 (darknet 기반의 weight만 이용한 OpenCV program)     
    ``` python opencv_images.py --image <test image 경로> --output <저장할 file 이름> --weights ./backup/yolov4_best.weights --config ./cfg/yolov4.cfg --classes ./data/obj.names ```

    ![opencv_test](https://user-images.githubusercontent.com/46422952/134079249-0d279e8a-3c29-481f-a57c-d9040626da67.jpg)

    <br>

2. Video     
    ```python darknet_video.py --input <test video 경로> --out_filename <저장할 file 이름> --weights ./backup/yolov4_best.weights --config_file ./cfg/yolov4.cfg --data_file ./data/obj.data --thresh 0.8 --ext_output ```    

    ![이미지](https://user-images.githubusercontent.com/46422952/134036997-a1aaa215-a125-4140-8b66-962de6fc82c4.GIF)      

<br>

***
<br>

## 실습 방법
: **Darknet 기반의 object detection의 경우, 자신의 PC에서 build한 darknet이 필요하므로 darknet을 설치하지 않을 경우 사용이 불가능합니다.**      
: **그러므로 실습을 위해 OpenCV 기반의 object detection을 사용하는 것을 권장드리며 이를 위해 필요한 file과 사용 예시를 적어두겠습니다.**     
: **예시의 경우, coco dataset에 대해 학습된 yolov4 model입니다.**   

1. File Download     
- opencv_image_annot.py download    
- coco.names download    
- https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights download    
- https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.cfg download


2. Run     
``` python opencv_image_annot.py --image <test image 경로> --output <저장할 file 이름> --weights <weights 확장자 file의 경로> --config <cfg 확장자 file의 경로> --classes <names 확장자 file 또는 class 이름들이 적힌 file의 경로> ```    
example)      
``` python opencv_image_annot.py --image test.jpg --output result.jpg --weights yolov4.weights --config yolov4.cfg --classes coco.names```    
(위의 예시가 정상적으로 동작하려면 모든 file이 동일한 folder에 위치해야합니다.)


<br>

***
<br>

## Reference
[1] Bochkovskiy, Alexey, Chien-Yao Wang, and Hong-Yuan Mark Liao. "Yolov4: Optimal speed and accuracy of object detection." arXiv preprint arXiv:2004.10934 (2020).    
[2] https://aiacademy.tw/yolo-v4-intro/    
[3] https://keyog.tistory.com/7    
[4] https://github.com/AlexeyAB/darknet     
[5] https://keyog.tistory.com/21?category=879585     
[6] https://eehoeskrap.tistory.com/370?category=705416      

<br>

***
<br>


## Contact
wldus8677@gmail.com    
   


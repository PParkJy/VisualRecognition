## dataset 폴더
training 폴더에서 사용하는 데이터셋입니다. 

<br/>

## trainig 폴더
본인이 가진 데이터셋으로 YOLO를 학습하고 싶을 때 참고하시면 됩니다.    
자세한 것은 training 폴더 하위의 README.md를 참고하세요. 

<br/>

## object_dectection.py
미리 학습된 YOLO를 사용하여 객체를 탐지하는 프로그램입니다.    
실행: <code> python object_detection.py --image dog.jpg --config yolov3.cfg --weights yolov3.weights --classes yolov3.txt  </code>    
실행결과: result.jpg

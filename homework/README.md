# 과제 부연설명 (읽어주세요!)

### Installation
- 프로그램 실행을 위해 **openCV 라이브러리**가 필요합니다. <code>pip install opencv-python</code> 명령어를 입력해서 설치해주세요.     
<br/>

### Download
1. https://github.com/PParkJy/VisualRecognition/ 사이트에서 example과 homework 파일을 다운로드 받습니다.
2. https://pjreddie.com/media/files/yolov3.weights 사이트에서 yolov3.weights 파일을 다운로드 받습니다. 
3. 다운받은 yolov3.weights 파일을 **homework.py가 위치한 폴더**로 이동시킵니다.
4. example 폴더에서 **yolov3.cfg, yolov3.txt** 파일을 **homework.py가 위치한 폴더**로 복사합니다.
5. yolov3.txt의 클래스들을 참고하여 인터넷에서 detection이 될 만한 이미지를 다운로드 합니다.     
   (예시: bicycle, cat이라는 클래스가 존재하므로 자전거와 고양이가 같이 있는 이미지를 다운로드)   
   **이미지 확장자는 jpg**를 추천합니다.
6. 다운받은 이미지를 **homework.py가 위치한 폴더**로 이동시킵니다.    
7. 6번 과정까지 완료하셨다면 **homework.py, yolov3.cfg, yolov3.txt, yolov3.weights, jpg 파일이 모두 같은 폴더에 위치해야 합니다.**

<br/>

### Execution
1. **homework.py를 주석을 참고하여 수정**합니다.
2. **cmd 창** (또는 anaconda prompt)를 열고 **homework.py가 위치한 폴더로 이동**합니다. (cd 명령어 사용) 
3. <code> python homework.py --image (본인이 다운받은 이미지 파일명.jpg) --config yolov3.cfg --weights yolov3.weights </code> 를 실행합니다. 
4. **result.jpg가 잘 생성된 것을 확인**합니다.

<br/>

### 과제 제출
- 과제는 **이클래스에 제출하도록 하며, 소스코드 파일명을 본인의 학번으로 바꾼 후 result.jpg와 함께 압축해서 제출**해주세요. 

import cv2
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--image', required=True,
                help = 'path to input image')
parser.add_argument('-c', '--config', required=True,
                help = 'path to yolo config file')
parser.add_argument('-w', '--weights', required=True,
                help = 'path to yolo pre-trained weights')
parser.add_argument('-cl', '--classes', required=True,
                help = 'path to text file containing class names')
parser.add_argument('-o', '--output', required=True,
                help = 'path to output image')

args = parser.parse_args()

def get_output_layers(net):
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers

def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = str(classes[class_id])
    color = COLORS[class_id]
    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)
    #cv2.rectangle(img, (x, y), (x + len(label) + 65, y - 25), color, 2)
    cv2.putText(img, label, (x+60,y-10), cv2.FONT_HERSHEY_SIMPLEX, 3, color, 2)


# Data load
image = cv2.imread(args.image)

Width = image.shape[1]
Width = image.shape[1] 
Height = image.shape[0]
scale = 0.00392

classes = None
with open(args.classes, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

# traoned YOLO model load
net = cv2.dnn.readNet(args.weights, args.config)

# image를 network에 넣기 위해 blob(특징 추출, 크기 조정)의 형태로 변환
# YOLO의 허용 크기: 320x320, 609X609, 416X416
blob = cv2.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True, crop=False)
net.setInput(blob)

# outs = 탐지된 객체에 대한 모든 정보와 위치 저장
outs = net.forward(get_output_layers(net))

# class_ids = class index
# box = bounding box의 좌표
# confidences = 객체의 신뢰도 (확률)

class_ids = []
confidences = []
boxes = []

# conf_threshold = 신뢰도
# 1에 가까울수록 정확도가 높으나 탐지되는 물체의 수는 적어짐
# 0에 가까울수록 정확도는 낮으나 탐지되는 물체의 수는 많아짐

conf_threshold = 0.5
nms_threshold = 0.4

for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            center_x = int(detection[0] * Width)
            center_y = int(detection[1] * Height)
            w = int(detection[2] * Width)
            h = int(detection[3] * Height)
            x = center_x - w / 2
            y = center_y - h / 2
            class_ids.append(class_id)
            confidences.append(float(confidence))
            boxes.append([x, y, w, h])



# 노이즈 제거
# 불필요한 bounding box 제거
indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

for i in indices:
    i = i[0]
    box = boxes[i]
    x = box[0]
    y = box[1]
    w = box[2]
    h = box[3]
    draw_prediction(image, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))
    print(f"conf: {confidences[i]} / x: {x} / y: {y} / width: {w} / height: {h}")

cv2.imwrite(args.output, image)
cv2.destroyAllWindows()
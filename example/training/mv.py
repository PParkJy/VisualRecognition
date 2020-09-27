import csv
import shutil

f1 = open("../ImageProcessing/example/dataset/train_labels.csv","r")
r = csv.reader(f1)

ard = []
hel = []
esp = []
ras = []

for i in r:
    if i[3] == "Arduino_Nano":
        ard.append(i[0].split(".")[0])
    elif i[3] == "Heltec_ESP32_Lora":
        hel.append(i[0].split(".")[0])
    elif i[3] == "ESP8266":
        esp.append(i[0].split(".")[0])
    else:
        ras.append(i[0].split(".")[0])

    for i in ard:
        try:
            shutil.move("../ImageProcessing/example/dataset/train/"+i+".jpg","../ImageProcessing/example/dataset/train/Arduino_Nano/")
            shutil.move("../ImageProcessing/example/dataset/train/"+i+".xml","../ImageProcessing/example/dataset/train/Arduino_Nano/")
        except:
            print("pass")
    for i in hel:
        try:
            shutil.move("../ImageProcessing/example/dataset/train/"+i+".jpg","../ImageProcessing/example/dataset/train/Heltec_ESP32_Lora/")
            shutil.move("../ImageProcessing/example/dataset/train/"+i+".xml","../ImageProcessing/example/dataset/train/Heltec_ESP32_Lora/")
        except:
            print("pass")
    for i in esp:
        try:
            shutil.move("../ImageProcessing/example/dataset/train/"+i+".jpg","../ImageProcessing/example/dataset/train/ESP8266/")
            shutil.move("../ImageProcessing/example/dataset/train/"+i+".xml","../ImageProcessing/example/dataset/train/ESP8266/")
        except:
            print("pass")
    for i in ras:
        try:
            shutil.move("../ImageProcessing/example/dataset/train/"+i+".jpg","../ImageProcessing/example/dataset/train/Raspberry_Pi_3/")
            shutil.move("../ImageProcessing/example/dataset/train/"+i+".xml","../ImageProcessing/example/dataset/train/Raspberry_Pi_3/")
        except:
            print("pass")
f1.close()
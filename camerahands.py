import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import cv2
import mediapipe as mp
from keras.models import load_model
import numpy as np
import time
import pyttsx3 # type: ignore

model = load_model('numbers.h5')

mphands = mp.solutions.hands
hands = mphands.Hands()
mp_drawing = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

_, frame = cap.read()

h, w, c = frame.shape

img_counter = 0
analysisframe = ''
letterpred = ['0','1','2','3','4','5','6','7','8','9'] 

# 初始化语音引擎
engine = pyttsx3.init()
# 设置语速
rate = engine.getProperty('rate')
engine.setProperty('rate', 125)
# 设置音量（范围为 0.0 到 1.0）
volume = engine.getProperty('volume')
engine.setProperty('volume', 1.0)

while True:
    # 检测按键输入，按下 ESC 键退出循环
    if cv2.waitKey(30) & 0xFF == 27:
        break
    
    # 检测有没有新的手势输入
    _, frame = cap.read()
    cv2.imshow("Frame",frame)
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(framergb)

    if result.multi_hand_landmarks:
        
        # 开始处理
        time.sleep(0.01)
        _, frame = cap.read()
        analysisframe = frame
        cv2.imshow("Frame", frame)
        framergbanalysis = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resultanalysis = hands.process(framergbanalysis)
        
        # 画图
        framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(framergb)
        hand_landmarks = result.multi_hand_landmarks
        for handLMs in hand_landmarks:
            x_max = 0
            y_max = 0
            x_min = w
            y_min = h
            for lm in handLMs.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                x_min = min(x_min, x)
                x_max = max(x_max, x)
                y_min = min(y_min, y)
                y_max = max(y_max, y)
            # 扩展边界框，避免超出图像边界
            y_min = max(0, y_min - 20)
            y_max = min(h, y_max + 20)
            x_min = max(0, x_min - 20)
            x_max = min(w, x_max + 20)
        
            for hand_landmarks in result.multi_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
        
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)    
        cv2.imshow("Frame", frame)
        
        # 计算    
        for handLMsanalysis in resultanalysis.multi_hand_landmarks:
            if x_min < x_max and y_min < y_max:  # 检查是否成功检测到手势并且坐标合法
                cropped_image = cv2.cvtColor(analysisframe, cv2.COLOR_BGR2GRAY)[y_min:y_max, x_min:x_max]
                resized = cv2.resize(cropped_image, (128, 128))  # 调整大小为128*128

                # 计算相对坐标
                relative_coords = []
                for lmanalysis in handLMsanalysis.landmark:
                    x_rel = int((lmanalysis.x * w - x_min) / (x_max - x_min) * 128)
                    y_rel = int((lmanalysis.y * h - y_min) / (y_max - y_min) * 128)
                    relative_coords.extend([x_rel, y_rel])

        prediction = model.predict(np.array([relative_coords]))  # 对相对坐标进行预测，注意将其转换为数组并添加一维
        predarray = prediction[0]  # 获取预测结果的第一个元素（假设预测结果是一个列表）
        letter_prediction_dict = {letterpred[i]: predarray[i] for i in range(len(letterpred))}  # 创建预测结果与类别的映射字典
        predarrayordered = sorted(predarray, reverse=True)  # 将预测值从高到低排序

        for high in predarrayordered[:1]:  # 取前一个最高预测值
            for key, value in letter_prediction_dict.items():
                if value == high:
                    print(f"Predicted Character {predarrayordered.index(high) + 1}: ", key)
                    print(f"Confidence {predarrayordered.index(high) + 1}: ", value)
                    break  # 输出完匹配的键值对后退出内层循环
        
        # 输出语音
        highest_confidence_character = max(letter_prediction_dict, key=letter_prediction_dict.get)
        engine.say(highest_confidence_character)
        engine.runAndWait()
        print("------------------------------------------")
    
cap.release()
cv2.destroyAllWindows()
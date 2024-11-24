import os
import json
import threading
import queue

import torch
from PIL import Image
from torchvision import transforms

import cv2
import mediapipe as mp
import numpy as np

from vit_model import vit_base_patch16_224_in21k as create_model

import pyttsx3

# 初始化 MediaPipe hands 和相机
mphands = mp.solutions.hands
hands = mphands.Hands()
cap = cv2.VideoCapture(0)
device = torch.device("cpu")

# 初始化语音引擎
engine = pyttsx3.init()
rate = engine.getProperty('rate')
engine.setProperty('rate', 125)
volume = engine.getProperty('volume')
engine.setProperty('volume', 1.0)

# 语音任务队列和锁
speech_queue = queue.Queue()
last_output = None
last_output_lock = threading.Lock()

# 手势缓冲区和锁
gesture_buffer = ""
buffer_lock = threading.Lock()

# 记录上一次和确认的手势
last_gesture = None
confirmed_gesture = None

def segment_hands_and_face(frame):
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    lower_skin = np.array([0, 133, 77], dtype=np.uint8)
    upper_skin = np.array([255, 173, 127], dtype=np.uint8)
    skin_mask = cv2.inRange(ycrcb, lower_skin, upper_skin)
    blurred = cv2.GaussianBlur(skin_mask, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((5, 5), np.uint8)
    morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(skin_mask)
    if len(contours) >= 3:
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        largest_contours = contours[:3]
        for contour in largest_contours:
            cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
    elif len(contours) > 0:
        for contour in contours:
            cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
    segmented = cv2.bitwise_and(frame, frame, mask=mask)
    return segmented

def speak_worker():
    while True:
        text = speech_queue.get()
        if text is None:
            break
        engine.say(text)
        engine.runAndWait()
        speech_queue.task_done()

def sm(strs):
    smlist = 'bpmfdtnlgkhjqxrzcsyw'
    nosm = ['eR', 'aN', 'eN', 'iN', 'uN', 'vN', 'nG', 'NG']
    rep = {'ZH': 'Zh', 'CH': 'Ch', 'SH': 'Sh'}
    for s in smlist:
        strs = strs.replace(s, s.upper())
    for s in nosm:
        strs = strs.replace(s, s.lower())
    for s in rep.keys():
        strs = strs.replace(s, rep[s])
    for s in nosm:
        tmp_num = 0
        isOk = False
        while (tmp_num < len(strs)) and (not isOk):
            try:
                tmp_num = strs.index(s.lower(), tmp_num)
            except ValueError:
                isOk = True
            else:
                tmp_num += len(s)
                if tmp_num < len(strs) and strs[tmp_num].lower() not in smlist:
                    strs = strs[:tmp_num - 1] + strs[tmp_num - 1].upper() + strs[tmp_num:]
    return strs

def onep(strs):
    restr = ''
    strs = sm(strs)
    for s in strs:
        if 'A' <= s <= 'Z':
            restr += ' ' + s
        else:
            restr += s
    restr = restr[1:]
    restr = restr.lower()
    return restr.split(' ')

def pinyin_2_hanzi(pinyinList):
    from Pinyin2Hanzi import DefaultDagParams
    from Pinyin2Hanzi import dag
    dagParams = DefaultDagParams()
    result = dag(dagParams, pinyinList, path_num=1, log=True)
    if result:
        res = result[0].path
        return res[0]
    return ''

def process_gesture(predict_cla, class_indict, output_current=False, clear_buffer=False):
    global gesture_buffer
    global last_gesture
    global confirmed_gesture
    global gesture_count
    
    if clear_buffer:
            with buffer_lock:
                gesture_buffer = ""
                print("缓冲区已清空")
            return

    if output_current:
        with buffer_lock:
            if gesture_buffer:
                pinyin_split = onep(gesture_buffer)
                if pinyin_split:
                    word = pinyin_split[0]
                    hanzi = pinyin_2_hanzi([word])
                    speech_queue.put(hanzi)
                    print("输出文字：", hanzi)
                    gesture_buffer = gesture_buffer[len(pinyin_split[0]):]
                    print(f"当前缓冲区（输出后）: {gesture_buffer}")  # 调试打印
        return
    
    gesture = class_indict[str(predict_cla)].lower()
    
    if gesture == last_gesture:
        gesture_count += 1
        if gesture_count == 2:
            if gesture == confirmed_gesture:
                return  # 忽略连续的相同手势
            confirmed_gesture = gesture
            with buffer_lock:
                # 检查缓冲区最后一个字符是否和当前手势相同
                if not gesture_buffer or gesture_buffer[-1] != gesture:
                    gesture_buffer += gesture
                    print(f"当前缓冲区（添加后）: {gesture_buffer}")  # 调试打印
                    pinyin_split = onep(gesture_buffer)
                    print(f"拼音： {pinyin_split}")  # 调试打印

                    if len(pinyin_split) > 1:
                        word = pinyin_split[0]
                        hanzi = pinyin_2_hanzi([word])
                        speech_queue.put(hanzi)
                        print("输出文字：", hanzi)
                        gesture_buffer = gesture_buffer[len(pinyin_split[0]):]
                        print(f"当前缓冲区（输出后）: {gesture_buffer}")  # 调试打印
            gesture_count = 0  # 重置计数器
    else:
        last_gesture = gesture
        gesture_count = 0  # 重置并初始化计数器

def main():
    gesture_count = 0
    data_transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    
    hands = mphands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    speech_thread = threading.Thread(target=speak_worker, daemon=True)
    speech_thread.start()

    global last_output
    global last_output_lock

    while True:
        if cv2.waitKey(30) & 0xFF == 27:
            break

        _, frame = cap.read()
        cv2.imshow("Camera", frame)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(frame_rgb)

        if result.multi_hand_landmarks:
            cv2.waitKey(10)
            _, img = cap.read()
            cv2.imshow("Camera", img)
            h, w, _ = img.shape
            resultanalysis = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            img = segment_hands_and_face(img)
            
            if resultanalysis.multi_hand_landmarks:
                try:
                    for handLMsanalysis in resultanalysis.multi_hand_landmarks:
                        x_max = 0
                        y_max = 0
                        x_min = w
                        y_min = h
                        x_vals = []
                        y_vals = []

                        for lmanalysis in handLMsanalysis.landmark:
                            x, y = int(lmanalysis.x * w), int(lmanalysis.y * h)
                            x_vals.append(x)
                            y_vals.append(y)

                        x_min, x_max = min(x_vals) - 20, max(x_vals) + 20
                        y_min, y_max = min(y_vals) - 20, max(y_vals) + 20

                        x_min = max(x_min, 0)
                        x_max = min(x_max, w)
                        y_min = max(y_min, 0)
                        y_max = min(y_max, h)

                        # 在图像上绘制边界框
                        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                        cv2.imshow("Camera", img)

                    if x_min < x_max and y_min < y_max:
                        img = img[y_min:y_max, x_min:x_max]
                        img = cv2.resize(img, (224, 224))

                    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                    img = data_transform(img)
                    img = torch.unsqueeze(img, dim=0)

                    json_path = 'class_indices.json'
                    assert os.path.exists(json_path), "文件: '{}' 不存在。".format(json_path)

                    with open(json_path, "r") as f:
                        class_indict = json.load(f)

                    model = create_model(num_classes=30, has_logits=False).to(device)
                    model_weight_path = "model-2.pth"
                    model.load_state_dict(torch.load(model_weight_path, map_location=device))
                    model.eval()

                    with torch.no_grad():
                        output = model(img.to(device))
                        predict = torch.softmax(output, dim=1)
                        predict_cla = torch.argmax(predict).item()
                        class_id = predict_cla
                        
                        # 输出预测结果的前三名
                        top5_prob, top5_cla = torch.topk(predict, 5)
                        print("-----------------------------------------------")
                        for i in range(3):
                            print(f"Top {i+1}: class: {class_indict[str(top5_cla[0][i].item())]}   prob: {top5_prob[0][i].item()}")

                        process_gesture(class_id, class_indict)

                        with last_output_lock:
                            if last_output != class_indict[str(predict_cla)]:
                                last_output = class_indict[str(predict_cla)]
                                process_gesture(predict_cla, class_indict)

                except Exception as e:
                    print(f"Error processing frame: {e}")

        key = cv2.waitKey(30) & 0xFF
        if key == 27:
            break
        elif key == ord('c'):
            process_gesture(None, None, clear_buffer=True)
        elif key == ord('f'):
            process_gesture(None, None, output_current=True)
                

    cap.release()
    cv2.destroyAllWindows()
    speech_queue.put(None)
    speech_thread.join()

if __name__ == "__main__":
    main()

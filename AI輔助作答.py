import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import pyperclip
import time

# 初始化 MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# 定義眼睛的特徵點索引（左眼和右眼，僅最上、最下、最左、最右）
LEFT_EYE_IDX = [159, 145, 133, 33]  # 左眼：上、下、左、右
RIGHT_EYE_IDX = [386, 374, 263, 362]  # 右眼：上、下、左、右

# 定義鼻子和嘴巴的特徵點索引
NOSE_IDX = 1
MOUTH_IDX = [13, 14]  # 嘴巴：上唇和下唇

# EAR 門檻值
EAR_THRESHOLD = 0.2
MOUTH_OPEN_THRESHOLD = 0.05

# 禁用 PyAutoGUI 的 FailSafe
pyautogui.FAILSAFE = False

# 操作冷卻時間
COOLDOWN_TIME = 1.5
last_action_time = 0

def calculate_ear(landmarks, eye_idx):
    # 計算眼睛的垂直和水平距離
    p1 = np.array([landmarks[eye_idx[0]].x, landmarks[eye_idx[0]].y])
    p2 = np.array([landmarks[eye_idx[1]].x, landmarks[eye_idx[1]].y])
    vertical_distance = np.linalg.norm(p1 - p2)
    
    p3 = np.array([landmarks[eye_idx[2]].x, landmarks[eye_idx[2]].y])
    p4 = np.array([landmarks[eye_idx[3]].x, landmarks[eye_idx[3]].y])
    horizontal_distance = np.linalg.norm(p3 - p4)
    
    # 計算 EAR（眼睛縱橫比）
    ear = vertical_distance / horizontal_distance
    return ear

def calculate_mouth_open(landmarks, mouth_idx):
    # 計算嘴巴的垂直距離
    p1 = np.array([landmarks[mouth_idx[0]].x, landmarks[mouth_idx[0]].y])
    p2 = np.array([landmarks[mouth_idx[1]].x, landmarks[mouth_idx[1]].y])
    vertical_distance = np.linalg.norm(p1 - p2)
    return vertical_distance

def draw_eye_landmarks(frame, landmarks, eye_idx):
    # 繪製眼睛的特徵點
    for idx in eye_idx:
        eye_point = landmarks[idx]
        eye_coords = (int(eye_point.x * frame.shape[1]), int(eye_point.y * frame.shape[0]))
        cv2.circle(frame, eye_coords, 3, (0, 255, 0), -1)

# 開啟攝影機
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

with mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # 翻轉影像（水平翻轉）
        frame = cv2.flip(frame, 1)
        # 轉換 BGR 影像為 RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # 進行人臉特徵點檢測
        results = face_mesh.process(rgb_frame)

        # 定義畫面區域寬度
        frame_height, frame_width, _ = frame.shape
        region_width = frame_width // 4

        # 設定區域顏色
        region_colors = [(50, 50, 50)] * 4  # 調暗背景
        region_labels = ['A', 'B', 'C', 'D']

        current_time = time.time()

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # 取得所有特徵點
                landmarks = face_landmarks.landmark

                # 計算左眼和右眼的 EAR（眼睛縱橫比）
                left_ear = calculate_ear(landmarks, LEFT_EYE_IDX)
                right_ear = calculate_ear(landmarks, RIGHT_EYE_IDX)

                # 判斷左眼是否閉眼
                if left_ear < EAR_THRESHOLD:
                    pyautogui.press('up')
                    cv2.putText(frame, 'Left Eye Closed', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                # 判斷右眼是否閉眼
                if right_ear < EAR_THRESHOLD:
                    pyautogui.press('down')
                    cv2.putText(frame, 'Right Eye Closed', (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                # 判斷嘴巴是否打開
                mouth_open = calculate_mouth_open(landmarks, MOUTH_IDX)
                if mouth_open > MOUTH_OPEN_THRESHOLD and (current_time - last_action_time) > COOLDOWN_TIME:
                    nose = landmarks[NOSE_IDX]
                    nose_coords = (int(nose.x * frame.shape[1]), int(nose.y * frame.shape[0]))
                    nose_x = nose_coords[0]
                    if 0 <= nose_x < frame_width:
                        region_index = min(nose_x // region_width, 3)
                        region_label = region_labels[region_index]
                        pyperclip.copy(region_label)
                        # 執行組合鍵操作
                        pyautogui.press('home')
                        pydirectinput.keyDown('shift')
                        pydirectinput.press('right')
                        pydirectinput.keyUp('shift')
                        time.sleep(0.05)
                        #pyautogui.press('delete')
                        pyautogui.hotkey('ctrl', 'v')
                        pyautogui.press('down')
                        last_action_time = current_time
                    cv2.putText(frame, 'Mouth Open', (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                # 繪製左眼和右眼的特徵點
                draw_eye_landmarks(frame, landmarks, LEFT_EYE_IDX)
                draw_eye_landmarks(frame, landmarks, RIGHT_EYE_IDX)

                # 獲取鼻子的位置並繪製
                nose = landmarks[NOSE_IDX]
                nose_coords = (int(nose.x * frame.shape[1]), int(nose.y * frame.shape[0]))
                cv2.circle(frame, nose_coords, 5, (255, 0, 0), -1)
                cv2.putText(frame, f'Nose: {nose_coords}', (nose_coords[0] + 10, nose_coords[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

                # 確定鼻子位於哪個區域，並將該區域反白
                if 0 <= nose_coords[0] < frame_width:
                    region_index = min(nose_coords[0] // region_width, 3)
                    region_colors[region_index] = (100, 100, 100)  # 反白顏色稍微亮一些

        # 繪製區域顏色
        for i in range(4):
            start_x = i * region_width
            end_x = (i + 1) * region_width
            frame[:, start_x:end_x] = cv2.addWeighted(frame[:, start_x:end_x], 0.7, np.full_like(frame[:, start_x:end_x], region_colors[i]), 0.3, 0)
            # 在每個區域中間顯示大字 A, B, C, D（使用白色顏色）
            text_x = start_x + region_width // 2 - 20
            text_y = frame_height // 2
            cv2.putText(frame, chr(65 + i), (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 5)

        # 顯示影像
        cv2.imshow('MediaPipe Face Mesh', frame)

        # 按下 'q' 鍵退出
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

# 釋放攝影機資源並關閉視窗
cap.release()
cv2.destroyAllWindows()

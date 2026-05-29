import cv2
import os

os.makedirs('video_frames', exist_ok=True)
cap = cv2.VideoCapture('fasterlive test .mp4')
count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imwrite(f'video_frames/f_{count:04d}.jpg', frame)
    count += 1
cap.release()
print(f"Extracted {count} frames.")

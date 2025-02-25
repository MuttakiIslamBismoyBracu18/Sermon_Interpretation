import cv2

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")

video_path = "E:\PRTS\Sample 1 - PRTS.mp4"  
cap = cv2.VideoCapture(video_path)

total_frames = 0
eyes_in_camera_frames = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    total_frames += 1

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face_roi = gray[y:y+h, x:x+w] 
        eyes = eye_cascade.detectMultiScale(face_roi, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20))

        if len(eyes) > 0:
            eyes_in_camera_frames += 1

        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(frame, (x+ex, y+ey), (x+ex+ew, y+ey+eh), (0, 255, 0), 2)

    cv2.imshow("Eye Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

eyes_in_camera_percent = (eyes_in_camera_frames / total_frames) * 100
eyes_not_in_camera_percent = 100 - eyes_in_camera_percent

print(f"Eyes in camera: {eyes_in_camera_percent:.2f}%")
print(f"Eyes not in camera: {eyes_not_in_camera_percent:.2f}%")
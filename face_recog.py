import cv2
import numpy as np
import face_recognition as fr

video_capture = cv2.VideoCapture(0 + cv2.CAP_DSHOW)

if video_capture.isOpened():
    print('Video beginning')

image = fr.load_image_file('sample_face.jpg')[0]
image_encoding = fr.face_encodings(image)

know_face_encodings = [
    image_encoding,
]

know_face_names = [
    "Khanh Truong",
]

while True:
    ret, frame = video_capture.read()

    rgb_frame = frame[:,:,::-1]

    fc_locations = fr.face_locations(rgb_frame)
    face_encodings = fr.face_encodings(rgb_frame, fc_locations)

    for (top, right, bottom, left), face_encoding in zip(fc_locations, face_encodings):
        matches = fr.compare_faces(know_face_encodings, face_encoding)

        name = "Unknown"

        fc_distance = fr.face_distance(know_face_encodings, face_encoding)
        best_match_index = np.argmin(fc_distance)

        if matches[best_match_index]:
            name = know_face_names[best_match_index]

        #Vẽ khung
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)

        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
    
    cv2.imshow('Face Recognition', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
import face_recognition as fr
import numpy as np
import cv2
import math
import os, sys

def face_confidence(face_distance, face_math_threshold=0.6):
    range = (1.0 - face_math_threshold)
    linear_val = (1.0 - face_distance) / (range * 2.0)

    if face_distance > face_math_threshold:
        return str(round(linear_val * 100, 2)) + '%'
    else:
        value = (linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2 - 0.2))) * 100
        return str(round(value, 2)) + '%'

class FaceRecognition:
    face_location = []
    face_encodings = []
    face_names = []
    known_faces_encodings = []
    known_faces_names = []
    process_current_frame = True

    def __init__(self):
        self.encode_faces()
        #encodings face
    
    def encode_faces(self):
        for image in os.listdir('faces'):
            face_image = fr.load_image_file(f'faces/{image}')
            face_encoding = fr.face_encodings(face_image)[0]

            self.known_faces_encodings.append(face_encoding)
            self.known_faces_names.append(image)
        
        print(self.known_faces_names)

    def run_recognition(self):
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            sys.exit('Video not found!')
        
        while True:
            ret, frame = cap.read()

            if not ret:
                sys.exit('Webcam not display')

            if self.process_current_frame:
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                rgb_small_frame = small_frame[:, :, ::-1]

                self.face_location = fr.face_locations(rgb_small_frame)
                self.face_encodings = fr.face_encodings(rgb_small_frame, self.face_location)

                self.face_names = []

                for face_encoding in self.face_encodings:
                    matches = fr.compare_faces(self.known_faces_encodings, face_encoding)
                    name = "Unknown"
                    confidence = "Unknown"

                    face_distances = fr.face_distance(self.face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)

                    if matches[best_match_index]:
                        name = self.known_faces_names[best_match_index]
                        confidence = face_confidence(face_distances[best_match_index])
                    
                    self.face_names.append(f'{name} ({confidence})')
            
            self.process_current_frame = not self.process_current_frame

            for (top, right, bottom, left), name in zip(self.face_locations, self.face_names):
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.rectangle(frame, (left, top -35), (right, bottom), (0, 0, 255), -1)
                cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255,255,255), 1)
            
            cv2.imshow('Face Recognition', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

fr = FaceRecognition()
fr.run_recognition()


import mediapipe as mp
import os
root = mp.__path__[0]
for dirpath, dirnames, filenames in os.walk(root):
    for f in filenames:
        if 'face' in f.lower() and (f.endswith('.tflite') or f.endswith('.task') or 'face_landmark' in f.lower() or 'face' in f.lower() and 'land' in f.lower()):
            print(os.path.join(dirpath, f))

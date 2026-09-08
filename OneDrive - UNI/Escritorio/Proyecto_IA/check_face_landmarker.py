from mediapipe.tasks.python.vision import face_landmarker
import inspect
print('face_landmarker attrs:', [a for a in dir(face_landmarker) if not a.startswith('_')])
print('\nFaceLandmarkerOptions:', inspect.getsource(face_landmarker.FaceLandmarkerOptions)[:1000])

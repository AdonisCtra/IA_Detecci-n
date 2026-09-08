import mediapipe as mp
import pkgutil
p = mp.tasks.python.vision
print('vision attrs:', [a for a in dir(p) if not a.startswith('_')])
print('vision pkg path:', p.__path__)
for finder, name, ispkg in pkgutil.iter_modules(p.__path__):
    print('module in vision:', name, 'ispkg=', ispkg)
try:
    import mediapipe.tasks.python.vision.face_mesh as fm
    print('face_mesh module attrs:', [a for a in dir(fm) if not a.startswith('_')])
except Exception as e:
    print('no face_mesh module via tasks API:', e)

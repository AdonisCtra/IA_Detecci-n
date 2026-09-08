import mediapipe as mp
import traceback

print('mediapipe version:', getattr(mp, '__version__', 'unknown'))
try:
    fm = mp.solutions.face_mesh.FaceMesh()
    print('FaceMesh init OK')
    fm.close()
except Exception as e:
    print('FaceMesh init ERROR:')
    traceback.print_exc()

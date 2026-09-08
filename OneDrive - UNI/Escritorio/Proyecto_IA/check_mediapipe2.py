import mediapipe as mp
import inspect

print('mediapipe file:', getattr(mp, '__file__', 'NO FILE'))
print('mediapipe package path:', getattr(mp, '__path__', 'NO PATH'))
print('attributes:', [a for a in dir(mp) if not a.startswith('_')])
# try to import solutions directly
try:
    import mediapipe.solutions as mps
    print('mediapipe.solutions import OK, attrs:', [a for a in dir(mps) if not a.startswith('_')])
except Exception as e:
    print('import mediapipe.solutions failed:', e)

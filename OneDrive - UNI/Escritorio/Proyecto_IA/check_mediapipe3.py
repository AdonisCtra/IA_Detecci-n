import mediapipe as mp
import pkgutil
import importlib

print('tasks attrs:', [a for a in dir(mp.tasks) if not a.startswith('_')])
# listar submodules en mediapipe.tasks
pkgpath = mp.tasks.__path__
print('tasks path:', pkgpath)
for finder, name, ispkg in pkgutil.iter_modules(pkgpath):
    print('module in tasks:', name, 'ispkg=', ispkg)
# intentar importar face mesh según nueva API
try:
    import mediapipe.tasks
    from mediapipe.tasks import python
    print('mediapipe.tasks.python attrs:', [a for a in dir(python) if not a.startswith('_')])
except Exception as e:
    print('no mediapipe.tasks.python:', e)

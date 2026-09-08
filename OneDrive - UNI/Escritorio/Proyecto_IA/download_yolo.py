"""Script para asegurar que `yolov8n.pt` esté en la raíz del proyecto.

Estrategia:
- Instancia `YOLO('yolov8n.pt')` para forzar la descarga por ultralytics.
- Busca en cachés comunes archivos llamados `yolov8n.pt` y copia el primero encontrado a la raíz.
"""
import os
import shutil
import sys
from pathlib import Path

def try_force_download():
    try:
        from ultralytics import YOLO
    except Exception as e:
        print("ultralytics no está instalado. Ejecuta: pip install ultralytics")
        raise

    try:
        print("Instanciando YOLO('yolov8n.pt') para forzar descarga (si es necesario)...")
        _ = YOLO('yolov8n.pt')
        print('Instanciación completada.')
    except Exception as e:
        print('Advertencia: no se pudo instanciar YOLO:', e)


def find_model(filename='yolov8n.pt'):
    # Buscar en ubicaciones probables
    home = Path.home()
    candidates = []
    search_paths = [
        Path('.').resolve(),
        home,
        home / '.cache',
        Path(sys.prefix) / 'Lib' / 'site-packages',
        Path(sys.prefix) / 'lib' / 'site-packages',
    ]

    for base in search_paths:
        if not base.exists():
            continue
        for root, dirs, files in os.walk(base):
            if filename in files:
                candidates.append(Path(root) / filename)
            # limitar búsqueda para no recorrer todo el home excesivamente
            if len(candidates) >= 3:
                break
        if candidates:
            break

    return candidates


def copy_to_project(src_path: Path, dest_filename='yolov8n.pt'):
    dest = Path(__file__).resolve().parent / dest_filename
    try:
        shutil.copy2(src_path, dest)
        print(f'Copiado {src_path} -> {dest}')
        return dest
    except Exception as e:
        print('Error copiando el archivo:', e)
        return None


def main():
    project_root = Path(__file__).resolve().parent
    dest = project_root / 'yolov8n.pt'
    if dest.exists():
        print('El modelo ya existe en la raíz:', dest)
        return

    try_force_download()

    candidates = find_model('yolov8n.pt')
    if not candidates:
        print('No se encontró yolov8n.pt en las rutas conocidas. Revisa la descarga automática de ultralytics o descarga manualmente.')
        return

    for c in candidates:
        res = copy_to_project(c)
        if res:
            break

    if not dest.exists():
        print('No se pudo copiar el modelo a la raíz. Descárgalo manualmente y colócalo en la carpeta del proyecto.')


if __name__ == '__main__':
    main()

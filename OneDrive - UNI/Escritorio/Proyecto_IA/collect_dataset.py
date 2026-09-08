import os
import time
import argparse
import cv2
from pathlib import Path

from celular import PhoneDetector


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description='Recolector manual de ejemplos (presiona p=phone, n=not-phone)')
    parser.add_argument('--out', type=str, default='dataset', help='Carpeta destino')
    parser.add_argument('--model', type=str, default='yolov8n.pt', help='Ruta modelo YOLO')
    parser.add_argument('--imgsz', type=int, default=320, help='Tamaño de entrada para detección')
    args = parser.parse_args()

    out_phone = Path(args.out) / 'phone'
    out_not = Path(args.out) / 'not_phone'
    ensure_dir(out_phone)
    ensure_dir(out_not)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print('No se pudo abrir la cámara')
        return

    try:
        detector = PhoneDetector(model_path=args.model)
    except Exception as e:
        print('Warning: PhoneDetector no disponible:', e)
        detector = None

    print('Controles: p = guardar como phone, n = guardar como not_phone, q = salir')
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        bbox = None
        if detector:
            try:
                res = detector.detect(frame, imgsz=args.imgsz)
                if res:
                    bbox = res['bbox']
                    x, y, w, h = bbox
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            except Exception:
                pass

        cv2.imshow('collect', frame)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break
        elif k == ord('p'):
            # guardar crop si hay bbox, si no guardar full frame
            ts = int(time.time() * 1000)
            if bbox:
                x, y, w, h = bbox
                crop = frame[y:y + h, x:x + w]
                path = out_phone / f'phone_{ts}.jpg'
            else:
                path = out_phone / f'phone_full_{ts}.jpg'
                crop = frame
            cv2.imwrite(str(path), crop)
            print('Guardado:', path)
        elif k == ord('n'):
            ts = int(time.time() * 1000)
            path = out_not / f'not_{ts}.jpg'
            cv2.imwrite(str(path), frame)
            print('Guardado not:', path)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

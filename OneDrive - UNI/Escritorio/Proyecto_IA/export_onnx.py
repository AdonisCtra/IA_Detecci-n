import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description='Exporta modelo YOLO (ultralytics) a ONNX')
    parser.add_argument('--model', type=str, default='yolov8n.pt', help='Ruta al modelo ultralytics')
    parser.add_argument('--imgsz', type=int, default=320, help='imgsz para export')
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except Exception as e:
        print('ultralytics no instalado:', e)
        sys.exit(1)

    print('Cargando modelo...', args.model)
    model = YOLO(args.model)
    print('Exportando a ONNX...')
    try:
        res = model.export(format='onnx', imgsz=args.imgsz)
        print('Export completo:', res)
    except Exception as e:
        print('Falló export ONNX:', e)


if __name__ == '__main__':
    main()

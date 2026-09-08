import cv2
from typing import Optional, Tuple

try:
	from ultralytics import YOLO
	_HAS_ULTRALYTICS = True
except Exception as e:
	_HAS_ULTRALYTICS = False
	_ULTRALYTICS_IMPORT_ERROR = e


class PhoneDetector:
	def __init__(self, model_path: str = 'yolov8n.pt', conf_threshold: float = 0.4):
		self.conf_threshold = conf_threshold
		self.model = None
		if not _HAS_ULTRALYTICS:
			raise RuntimeError(
				"Ultralytics YOLO no está instalado. Ejecuta: pip install ultralytics\n" +
				f"Error original: {_ULTRALYTICS_IMPORT_ERROR}")
		try:
			# YOLO constructor descarga el modelo si es necesario
			self.model = YOLO(model_path)
		except Exception as e:
			raise RuntimeError(f"No se pudo cargar el modelo YOLO desde {model_path}: {e}")

	def detect(self, frame, imgsz: int = 640) -> Optional[dict]:
		"""Detecta si hay un teléfono en el frame. Retorna None o dict con bbox y conf.
		Parámetro `imgsz` permite controlar el tamaño de entrada para acelerar inferencia.
		"""
		if not self.model:
			return None
		results = self.model(frame, imgsz=imgsz, conf=self.conf_threshold)
		if not results:
			return None
		r = results[0]
		names = getattr(self.model, 'names', {}) or {}

		# Intentar extraer arrays de boxes, clases y confs de forma segura
		try:
			boxes = r.boxes.xyxy.tolist()
			classes = r.boxes.cls.tolist()
			confs = r.boxes.conf.tolist()
		except Exception:
			# Si no se puede, devolvemos None (compatibilidad futura)
			return None

		for box, cls, conf in zip(boxes, classes, confs):
			try:
				cls_idx = int(cls)
			except Exception:
				cls_idx = None
			if isinstance(names, dict):
				cls_name = names.get(cls_idx, str(cls_idx))
			elif isinstance(names, (list, tuple)):
				try:
					cls_name = names[cls_idx]
				except Exception:
					cls_name = str(cls_idx)
			else:
				cls_name = str(cls_idx)

			# Buscar etiquetas que indiquen teléfono (ej: 'cell phone', 'mobile phone', 'phone')
			if 'phone' in cls_name.lower() or 'cell' in cls_name.lower() or 'mobile' in cls_name.lower():
				x1, y1, x2, y2 = map(int, box)
				return {"bbox": (x1, y1, x2 - x1, y2 - y1), "conf": float(conf), "class": cls_name}
		return None


if __name__ == "__main__":
	cap = cv2.VideoCapture(0)
	det = PhoneDetector(conf_threshold=0.35)
	while True:
		ret, frame = cap.read()
		if not ret:
			break
		res = det.detect(frame)
		if res:
			x, y, w, h = res['bbox']
			cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
			cv2.putText(frame, f"{res['class']} {res['conf']:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
		cv2.imshow('phone test', frame)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	cap.release()
	cv2.destroyAllWindows()


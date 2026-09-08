import cv2
import numpy as np
from typing import Optional, Tuple

try:
	import mediapipe as mp
	mp_face = mp.solutions.face_mesh
	_HAS_MEDIAPIPE = True
except Exception:
	mp = None
	mp_face = None
	_HAS_MEDIAPIPE = False


def _eye_aspect_ratio(eye_landmarks: np.ndarray) -> float:
	# eye_landmarks: array of shape (6,2) with the standard 6-eye points
	A = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
	B = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
	C = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
	if C == 0:
		return 0.0
	ear = (A + B) / (2.0 * C)
	return float(ear)


def _estimate_head_pose(lm_coords: np.ndarray, frame_size: Tuple[int, int]):
	"""Estima yaw/pitch/roll en grados usando solvePnP con puntos faciales comunes.
	lm_coords: array Nx2 de landmarks en pixeles
	Devuelve (yaw_deg, pitch_deg, roll_deg) o (None, None, None) si falla.
	"""
	try:
		# Puntos 2D en la imagen: usar índices conocidos de MediaPipe FaceMesh
		image_points = []
		# nariz (tip)
		image_points.append(tuple(lm_coords[1]))
		# mentón
		image_points.append(tuple(lm_coords[152]))
		# esquina izquierda del ojo (outer)
		image_points.append(tuple(lm_coords[33]))
		# esquina derecha del ojo (outer)
		image_points.append(tuple(lm_coords[263]))
		# esquina izquierda de la boca
		image_points.append(tuple(lm_coords[61]))
		# esquina derecha de la boca
		image_points.append(tuple(lm_coords[291]))

		image_points = np.array(image_points, dtype='double')

		# Modelo 3D aproximado de estos puntos en mm (valor tomado de ejemplos comunes)
		model_points = np.array([
			(0.0, 0.0, 0.0),         # nariz
			(0.0, -63.6, -12.5),     # mentón
			(-43.3, 32.7, -26.0),    # left eye corner
			(43.3, 32.7, -26.0),     # right eye corner
			(-28.9, -28.9, -24.1),   # left mouth
			(28.9, -28.9, -24.1)     # right mouth
		], dtype='double')

		h, w = frame_size
		focal_length = w
		center = (w / 2, h / 2)
		camera_matrix = np.array([
			[focal_length, 0, center[0]],
			[0, focal_length, center[1]],
			[0, 0, 1]
		], dtype='double')

		dist_coeffs = np.zeros((4, 1))

		success, rotation_vec, translation_vec = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
		if not success:
			return (None, None, None)

		rot_mat, _ = cv2.Rodrigues(rotation_vec)
		proj_matrix = np.hstack((rot_mat, translation_vec))
		# Decompose to yaw/pitch/roll
		_, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(proj_matrix)
		pitch, yaw, roll = euler_angles.flatten()
		# cv2 returns pitch, yaw, roll in degrees
		# Asegurar float
		return (float(yaw), float(pitch), float(roll))
	except Exception:
		return (None, None, None)


class FaceAttention:
	def __init__(self, max_num_faces: int = 1, refine_landmarks: bool = False):
		if not _HAS_MEDIAPIPE:
			self.face_mesh = None
			return
		# Intentar inicializar FaceMesh; algunas versiones de mediapipe no soportan
		# el argumento `refine_landmarks`, así que probamos con y sin él y
		# exponemos el error si no funciona.
		try:
			self.face_mesh = mp_face.FaceMesh(static_image_mode=False,
							  max_num_faces=max_num_faces,
							  refine_landmarks=refine_landmarks,
							  min_detection_confidence=0.5,
							  min_tracking_confidence=0.5)
		except TypeError:
			# intentar sin refine_landmarks
			try:
				self.face_mesh = mp_face.FaceMesh(static_image_mode=False,
								max_num_faces=max_num_faces,
								min_detection_confidence=0.5,
								min_tracking_confidence=0.5)
			except Exception as e:
				print('Error inicializando MediaPipe FaceMesh:', e)
				self.face_mesh = None
		except Exception as e:
			print('Error inicializando MediaPipe FaceMesh:', e)
			self.face_mesh = None

	def process(self, frame: np.ndarray) -> Optional[dict]:
		# Devuelve None si no hay cara, o dict con métricas
		if self.face_mesh is None:
			return None
		img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		results = self.face_mesh.process(img_rgb)
		if not results or not results.multi_face_landmarks:
			return None

		lm = results.multi_face_landmarks[0]
		h, w = frame.shape[:2]

		# Índices de referencia (MediaPipe Face Mesh)
		# ojo izquierdo (approx): [33, 160, 158, 133, 153, 144]
		# ojo derecho (approx): [362, 385, 387, 263, 373, 380]
		left_idx = [33, 160, 158, 133, 153, 144]
		right_idx = [362, 385, 387, 263, 373, 380]

		lm_coords = np.array([[p.x * w, p.y * h] for p in lm.landmark])

		left_eye = lm_coords[left_idx]
		right_eye = lm_coords[right_idx]

		left_ear = _eye_aspect_ratio(left_eye)
		right_ear = _eye_aspect_ratio(right_eye)
		ear = (left_ear + right_ear) / 2.0

		# Coordenadas de nariz (tip) y punto medio entre ojos para estimar desviación horizontal
		nose_idx = 1  # punta de la nariz
		left_eye_center = left_eye.mean(axis=0)
		right_eye_center = right_eye.mean(axis=0)
		eyes_mid = (left_eye_center + right_eye_center) / 2.0
		nose = lm_coords[nose_idx]

		# Estimar orientación de la cabeza (yaw/pitch/roll)
		head_yaw, head_pitch, head_roll = _estimate_head_pose(lm_coords, (h, w))

		# Normalizar desplazamiento horizontal por el ancho de la cara (distancia entre ojos)
		eye_distance = np.linalg.norm(left_eye_center - right_eye_center)
		dx = (nose[0] - eyes_mid[0]) / (eye_distance + 1e-6)

		# Caja aproximada de la cara
		x_min = int(lm_coords[:, 0].min())
		y_min = int(lm_coords[:, 1].min())
		x_max = int(lm_coords[:, 0].max())
		y_max = int(lm_coords[:, 1].max())
		bbox = (x_min, y_min, x_max - x_min, y_max - y_min)

		return {
			"ear": ear,
			"nose_dx": dx,
			"head_yaw": head_yaw,
			"head_pitch": head_pitch,
			"head_roll": head_roll,
			"bbox": bbox,
			"landmarks": lm_coords,
		}

	def close(self):
		if self.face_mesh is not None:
			try:
				self.face_mesh.close()
			except Exception:
				pass


if __name__ == "__main__":
	# Pequeña prueba manual si se ejecuta directo
	cap = cv2.VideoCapture(0)
	detector = FaceAttention()
	if detector.face_mesh is None:
		print("MediaPipe no está disponible. Instala 'mediapipe' para probar este módulo.")
	while True:
		ret, frame = cap.read()
		if not ret:
			break
		info = detector.process(frame)
		if info:
			x, y, w, h = info["bbox"]
			cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
			cv2.putText(frame, f"EAR:{info['ear']:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
		cv2.imshow("test rostro", frame)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	cap.release()
	detector.close()
	cv2.destroyAllWindows()


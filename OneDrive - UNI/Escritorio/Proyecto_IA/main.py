import time
import sys
import cv2
import platform
import threading
import queue
import argparse
import statistics
import json
import os
from datetime import datetime
from pathlib import Path

try:
	from gpu_utils import get_device
except Exception:
	def get_device():
		return 'unknown'

from config import *
from rostro import FaceAttention
from celular import PhoneDetector
from audio_utils import ensure_alert_sound, play_alert
from collections import deque
from events import EventRecorder

try:
	import winsound
	_HAS_WINSOUND = True
except Exception:
	_HAS_WINSOUND = False


def alert():
	# Intentar reproducir archivo de alerta, si no usar winsound
	sound_path = ensure_alert_sound()
	played = play_alert(path=sound_path, freq=ALERT_FREQUENCY, duration_ms=ALERT_DURATION_MS)
	if not played:
		print("ALERTA: distracción detectada!")


def main():
	parser = argparse.ArgumentParser(description='Monitor de distracciones')
	parser.add_argument('--duration', type=float, default=0.0, help='Duración máxima en segundos (0=infinito)')
	parser.add_argument('--no-audio', action='store_true', help='Desactivar audio de alerta')
	parser.add_argument('--no-save', action='store_true', help='No guardar clips de eventos')
	parser.add_argument('--model', type=str, default=PHONE_MODEL_PATH, help='Ruta al modelo YOLO')
	parser.add_argument('--imgsz', type=int, default=PHONE_IMG_SIZE, help='Tamaño de entrada para YOLO')
	parser.add_argument('--head-yaw-threshold', type=float, default=None, help='Umbral yaw (grados) para considerar giro de cabeza')
	parser.add_argument('--verifier', type=str, default=None, help='Ruta al modelo verificador (.pth) para validar crops')
	parser.add_argument('--verifier-threshold', type=float, default=0.5, help='Probabilidad mínima para aceptar crop como phone')
	parser.add_argument('--calibrate', action='store_true', help='Modo calibración: medir yaw neutro y proponer umbral')
	parser.add_argument('--calibrate-duration', type=float, default=8.0, help='Segundos para la calibración')
	parser.add_argument('--debug', action='store_true', help='Habilitar logs verbosos y detectar teléfono cada frame')
	parser.add_argument('--save-frames', action='store_true', help='Permitir guardar vídeo/miniaturas de eventos')
	parser.add_argument('--retention-days', type=int, default=EVENT_RETENTION_DAYS, help='Días de retención de eventos')
	parser.add_argument('--max-event-files', type=int, default=EVENT_MAX_FILES, help='Máximo de archivos de eventos')
	parser.add_argument('--no-encrypt-events', action='store_true', help='Desactivar cifrado de metadatos')
	args = parser.parse_args()

	# Abrir cámara; si falla intentar índices alternativos
	cap = cv2.VideoCapture(CAMERA_INDEX)
	if not cap.isOpened():
		for ci in range(1, 5):
			cap = cv2.VideoCapture(ci)
			if cap.isOpened():
				print(f'Usando cámara alternativa index={ci}')
				break
		if not cap.isOpened():
			print("No se pudo abrir la cámara. Revisa CAMERA_INDEX en config.py o que la cámara esté conectada.")
			sys.exit(1)

	face = FaceAttention()
	if face.face_mesh is None:
		print('Aviso: FaceAttention no inicializó MediaPipe FaceMesh. Revisa instalación de mediapipe o permisos de cámara.')
	phone = PhoneDetector(model_path=args.model or PHONE_MODEL_PATH, conf_threshold=PHONE_CONF_THRESHOLD)

	# Cargar verificador secundario si está disponible
	verifier = None
	verifier_threshold = float(args.verifier_threshold)
	if args.verifier:
		try:
			from classifier import Verifier
			verifier = Verifier(args.verifier, device='cuda' if (get_device()=='cuda') else 'cpu')
			print('Verificador cargado desde', args.verifier)
		except Exception as e:
			print('No se pudo cargar verificador:', e)
	else:
		# intentar cargar classifier.pth en la raíz si existe
		if Path('classifier.pth').exists():
			try:
				from classifier import Verifier
				verifier = Verifier('classifier.pth', device='cuda' if (get_device()=='cuda') else 'cpu')
				print('Verificador cargado desde classifier.pth')
			except Exception as e:
				print('No se pudo cargar classifier.pth:', e)

	# Cola y señal de terminación para el worker de inferencia
	frame_queue = queue.Queue(maxsize=4)
	stop_event = threading.Event()
	shared_lock = threading.Lock()
	shared = {
		'face': None,
		'phone': None,
		'timestamp': 0.0,
	}

	def inference_worker(frame_q: queue.Queue, shared_state: dict, lock: threading.Lock, stop_evt: threading.Event):
		phone_frame_counter = 0
		debug_flag = args.debug if 'args' in locals() or 'args' in globals() else False
		while not stop_evt.is_set():
			try:
				frame = frame_q.get(timeout=0.25)
			except queue.Empty:
				continue

			# Procesar rostro (MediaPipe)
			try:
				info = face.process(frame) if face and getattr(face, 'face_mesh', None) is not None else None
			except Exception:
				info = None
			if debug_flag and info:
				try:
					print(f'[DEBUG][face] EAR={info.get("ear"):.3f} head_yaw={info.get("head_yaw")} bbox={info.get("bbox")}')
				except Exception:
					print('[DEBUG][face] info disponible')

			# Detección de teléfono con intervalo para aligerar carga
			phone_res = None
			phone_frame_counter += 1
			try:
				interval = PHONE_DETECT_INTERVAL
			except NameError:
				interval = 3
			if phone_frame_counter >= interval:
				phone_frame_counter = 0
				try:
					phone_res = phone.detect(frame, imgsz=PHONE_IMG_SIZE) if phone and getattr(phone, 'model', None) else None
					# Verificación secundaria: crop + classifier
					if phone_res and verifier is not None:
						x, y, w, h = phone_res['bbox']
						# asegurar límites
						h0, w0 = frame.shape[:2]
						x1 = max(0, x)
						y1 = max(0, y)
						x2 = min(w0, x + w)
						y2 = min(h0, y + h)
						if x2 > x1 and y2 > y1:
							crop = frame[y1:y2, x1:x2]
							try:
								crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
								score = verifier.predict_prob(crop_rgb)
								phone_res['verifier_score'] = float(score)
								phone_res['verified'] = bool(score >= verifier_threshold)
							except Exception:
								phone_res['verifier_score'] = None
								phone_res['verified'] = False
					# Debug prints
					if debug_flag and phone_res:
						try:
							print(f"[DEBUG][phone] bbox={phone_res.get('bbox')} conf={phone_res.get('conf')} verified={phone_res.get('verified')} verifier_score={phone_res.get('verifier_score')}")
						except Exception:
							print('[DEBUG][phone] phone_res disponible')
					# Guardado automático de false positives (throttle local)
					try:
						if phone_res and verifier is not None and not phone_res.get('verified', False) and float(phone_res.get('conf', 0.0)) >= PHONE_CONF_THRESHOLD:
							now_fp = time.time()
							if not hasattr(inference_worker, 'last_fp_save') or (now_fp - getattr(inference_worker, 'last_fp_save', 0.0)) > 5.0:
								# crear carpeta
								fp_dir = Path('events') / 'false_positives'
								fp_dir.mkdir(parents=True, exist_ok=True)
								ts = datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')
								img_path = fp_dir / f'fp_{ts}.jpg'
								try:
									cv2.imwrite(str(img_path), crop)
									meta = {
										'timestamp_utc': datetime.utcnow().isoformat() + 'Z',
										'phone_conf': float(phone_res.get('conf', 0.0)),
										'verifier_score': phone_res.get('verifier_score'),
										'notes': 'auto false positive'
									}
									with open(str(img_path.with_suffix('.json')), 'w', encoding='utf-8') as mf:
										json.dump(meta, mf, ensure_ascii=False, indent=2)
										inference_worker.last_fp_save = now_fp
								except Exception:
									pass
					except Exception:
						# ignorar errores de guardado, no bloquear inferencia
						pass
				except Exception:
					phone_res = None

			with lock:
				shared_state['face'] = info
				shared_state['phone'] = phone_res
				shared_state['timestamp'] = time.time()

		# limpieza final si es necesario
		return

	# Si debug, forzar detección de teléfono cada frame para pruebas
	if args.debug:
		try:
			PHONE_DETECT_INTERVAL = 1
		except Exception:
			pass
	worker = threading.Thread(target=inference_worker, args=(frame_queue, shared, shared_lock, stop_event), daemon=False)
	worker.start()

	# Head yaw threshold: prefer CLI, si no usar config
	head_yaw_threshold = args.head_yaw_threshold if args.head_yaw_threshold is not None else HEAD_YAW_THRESHOLD_DEG

	# Calibración asistida (si se solicita)
	if args.calibrate:
		print(f'Iniciando calibración de head yaw durante {args.calibrate_duration}s...')
		samples = []
		cal_start = time.time()
		while time.time() - cal_start < args.calibrate_duration:
			ret, calibration_frame = cap.read()
			if not ret:
				break
			try:
				frame_queue.put_nowait(calibration_frame.copy())
			except queue.Full:
				pass
			with shared_lock:
				inf = shared.get('face')
			if inf and inf.get('head_yaw') is not None:
				try:
					samples.append(float(inf.get('head_yaw')))
				except Exception:
					pass
			remaining = max(0.0, args.calibrate_duration - (time.time() - cal_start))
			cv2.rectangle(calibration_frame, (10, 10), (620, 92), (25, 25, 25), -1)
			cv2.putText(calibration_frame, 'CALIBRACION: mira al frente y mantente quieto',
						(20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 220, 220), 2)
			cv2.putText(calibration_frame, f'Tiempo restante: {remaining:.1f}s | muestras: {len(samples)} | q: cancelar',
						(20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (235, 235, 235), 1)
			if SHOW_DEBUG_WINDOW:
				cv2.imshow('Monitor Distracciones', calibration_frame)
				if (cv2.waitKey(1) & 0xFF) == ord('q'):
					print('Calibración cancelada por el usuario.')
					break
			time.sleep(0.03)

		if samples:
			mean = statistics.mean([abs(x) for x in samples])
			stdev = statistics.pstdev([abs(x) for x in samples])
			suggested = max(HEAD_YAW_THRESHOLD_DEG, mean + 2 * stdev)
			head_yaw_threshold = suggested
			print(f'Calibración completada: mean={mean:.2f}°, std={stdev:.2f}°. Umbral sugerido={suggested:.2f}°')

			# Guardar estadísticas de calibración en JSON (atomically)
			try:
				events_dir = Path('events')
				events_dir.mkdir(parents=True, exist_ok=True)
				ts = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
				base = events_dir / f'calibration_{ts}'
				meta = {
					'timestamp_utc': datetime.utcnow().isoformat() + 'Z',
					'duration_s': float(args.calibrate_duration),
					'sample_count': len(samples),
					'mean_yaw_deg': mean,
					'std_yaw_deg': stdev,
					'suggested_threshold_deg': suggested,
					'cli_args': {
						'head_yaw_threshold': args.head_yaw_threshold,
						'imgsz': args.imgsz,
						'model': args.model,
						'duration': args.duration,
						'no_audio': args.no_audio,
						'no_save': args.no_save
					},
					'device': get_device(),
					'notes': 'Auto-calibration from session.'
				}
				meta_path_tmp = str(base.with_suffix('.json.tmp'))
				meta_path = str(base.with_suffix('.json'))
				with open(meta_path_tmp, 'w', encoding='utf-8') as f:
					json.dump(meta, f, ensure_ascii=False, indent=2)
				os.replace(meta_path_tmp, meta_path)
				print('Calibración guardada en', meta_path)
			except Exception as e:
				print('No se pudo guardar calibración:', e)
		else:
			print('No se recolectaron muestras de yaw durante la calibración; usando umbral por defecto.')

	closed_eyes_frames = 0
	distraction_start = None
	# Usar valor configurable del archivo de configuración
	try:
		timeout_to_alert = DISTRACTION_SECONDS_TO_ALERT
	except NameError:
		timeout_to_alert = 2.0

	# Preparar buffer circular para guardar frames recientes (para clips)
	fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
	try:
		fps = float(fps) if fps > 0 else 20.0
	except Exception:
		fps = 20.0
	buffer_seconds = 5.0
	post_seconds = 3.0
	max_buffer_frames = int(buffer_seconds * fps)
	frame_buffer = deque(maxlen=max_buffer_frames)
	recorder = EventRecorder(out_dir='events', log_file='events/events.log',
	                         encrypt_metadata=EVENT_ENCRYPT_METADATA and not args.no_encrypt_events)
	try:
		recorder.cleanup(retention_days=args.retention_days, max_files=args.max_event_files)
	except Exception as exc:
		print(f'Aviso: no se pudo limpiar eventos antiguos: {exc}')
	last_cleanup = time.time()
	last_event_time = None
	pending_save = None
	# Contadores para detección de teléfono por N cuadros consecutivos
	phone_confirm_count = 0
	phone_frame_counter = 0

	# Estructuras para suavizado temporal
	phone_hits = deque()
	face_hits = deque()

	start_time = time.time()
	max_duration = float(args.duration or 0.0)

	while True:
		ret, frame = cap.read()
		if not ret:
			break

		# Encolar frame para inferencia (no bloquear la UI). Si la cola está llena, descartar el más antiguo.
		try:
			frame_queue.put_nowait(frame.copy())
		except queue.Full:
			try:
				_ = frame_queue.get_nowait()
			except Exception:
				pass
			try:
				frame_queue.put_nowait(frame.copy())
			except Exception:
				pass

		# Leer resultados de inferencia compartidos por el worker
		with shared_lock:
			info = shared.get('face')
			phone_res = shared.get('phone')
			res_ts = shared.get('timestamp', 0.0)

		# Mostrar detección de teléfono (estado de verificación) si existe
		if phone_res:
			try:
				x, y, w, h = phone_res['bbox']
				color = (0, 0, 255) if phone_res.get('verified') else (0, 165, 255)
				cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
				lbl = f"PHONE {phone_res.get('conf',0.0):.2f}"
				vs = phone_res.get('verifier_score', None)
				if vs is not None:
					lbl += f" v:{vs:.2f}"
				cv2.putText(frame, lbl, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
			except Exception:
				pass

		now = time.time()

		# Actualizar historiales temporales
		window = DETECTION_WINDOW_SECONDS
		if phone_res:
			# si hay verificador, sólo contar hits verificados
			try:
				if verifier is None or bool(phone_res.get('verified', False)):
					phone_hits.append(now)
			except Exception:
				phone_hits.append(now)
		# Si info indica posible distracción (ojos cerrados o cabeza girada), registrar hit
		if info:
			try:
				# ojos cerrados o nariz desviada
				if info.get('ear', 1.0) < EYE_AR_THRESHOLD or abs(info.get('nose_dx', 0.0)) > HEAD_TURN_THRESHOLD:
					face_hits.append(now)
				else:
					# comprobar orientación de la cabeza (yaw en grados) con el umbral calibrado/seleccionado
					hy = info.get('head_yaw', None)
					if hy is not None:
						try:
							if abs(float(hy)) > head_yaw_threshold:
								face_hits.append(now)
						except Exception:
							pass
			except Exception:
				pass

		# Limpiar hits fuera de la ventana temporal
		while phone_hits and (now - phone_hits[0]) > window:
			phone_hits.popleft()
		while face_hits and (now - face_hits[0]) > window:
			face_hits.popleft()

		phone_confirmed = len(phone_hits) >= PHONE_REQUIRED_IN_WINDOW
		face_confirmed = len(face_hits) >= FACE_REQUIRED_IN_WINDOW

		# Construir razones
		reasons = []
		if face_confirmed:
			reasons.append('face')
		if phone_confirmed:
			reasons.append('phone')

		distracted = False

		if info:
			ear = info['ear']
			nose_dx = info['nose_dx']
			if ear < EYE_AR_THRESHOLD:
				closed_eyes_frames += 1
			else:
				closed_eyes_frames = 0

			if closed_eyes_frames >= EYE_AR_CONSEC_FRAMES:
				distracted = True

			if abs(nose_dx) > HEAD_TURN_THRESHOLD:
				distracted = True

			x, y, w, h = info['bbox']
			cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
			cv2.putText(frame, f"EAR:{ear:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
			# Mostrar head_yaw si está disponible
			hy = info.get('head_yaw', None)
			if hy is not None:
				try:
					cv2.putText(frame, f"Yaw:{float(hy):.1f}°", (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 200), 2)
				except Exception:
					pass

		# Requerir detecciones por ventana temporal para considerar teléfono como distracción
		if phone_confirmed:
			distracted = True
			if phone_res:
				x, y, w, h = phone_res['bbox']
				cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
				cv2.putText(frame, f"PHONE {phone_res['conf']:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

		if distracted:
			if distraction_start is None:
				distraction_start = time.time()
			elif time.time() - distraction_start >= timeout_to_alert:
				now = time.time()
				# Evitar guardar/alertar repetidamente: respetar cooldown entre eventos
				try:
					cooldown = EVENT_COOLDOWN_SECONDS
				except NameError:
					cooldown = 10.0

				if last_event_time is None or (now - last_event_time) >= cooldown:
					# Preparar guardado sin bloquear: copiar buffer y recoger post-frames en el bucle
					if not args.no_save and (args.save_frames or SAVE_EVENT_FRAMES):
						pending_save = {
							'frames': list(frame_buffer),
							'post_needed': int(post_seconds * fps),
							'fps': fps,
							'label': 'distraccion',
							'reasons': reasons
						}
					elif not args.no_save:
						try:
							recorder.save_metadata(label='distraccion', reasons=reasons, fps=fps)
						except Exception as exc:
							print(f'Aviso: no se pudo registrar el evento: {exc}')
					# Alertar inmediatamente y marcar tiempo
					if not args.no_audio:
						alert()
					last_event_time = now
				distraction_start = None
		else:
			distraction_start = None

		# Añadir frame al buffer antes de mostrar/analizar
		frame_buffer.append(frame.copy())

		# Si hay un pendiente para guardar, recolectar post-frames aquí (sin bloquear)
		if pending_save is not None:
			pending_save['frames'].append(frame.copy())
			pending_save['post_needed'] -= 1
			if pending_save['post_needed'] <= 0:
				# Lanzar hilo para guardar las frames
				frames_to_save = pending_save['frames'].copy()
				# Capturar valores en variables locales para evitar condiciones de carrera
				fps_local = pending_save.get('fps')
				label_local = pending_save.get('label')
				reasons_local = pending_save.get('reasons', []) if pending_save is not None else []
				def _bg_save(frames, fps, label, reasons):
					recorder.save_frames(frames, fps=fps, label=label, reasons=reasons)
				t = threading.Thread(target=_bg_save, args=(frames_to_save, fps_local, label_local, reasons_local), daemon=True)
				t.start()
				print('Guardado en background lanzado')
				pending_save = None

		# Indicador compacto de estado para saber qué está ocurriendo sin mirar la consola.
		camera_status = 'CAM OK'
		face_status = 'ROSTRO OK' if info else 'ROSTRO --'
		phone_status = 'TEL OK' if phone_res else 'TEL --'
		attention_status = 'ALERTA' if distracted else 'ATENTO'
		status_color = (0, 0, 255) if distracted else (0, 180, 0)
		cv2.rectangle(frame, (8, 8), (300, 42), (25, 25, 25), -1)
		cv2.putText(frame, f'{camera_status} | {face_status} | {phone_status}',
					(14, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (230, 230, 230), 1)
		cv2.putText(frame, attention_status, (14, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.42, status_color, 1)
		if time.time() - last_cleanup >= EVENT_CLEANUP_INTERVAL_SECONDS:
			try:
				recorder.cleanup(retention_days=args.retention_days, max_files=args.max_event_files)
			except Exception:
				pass
			last_cleanup = time.time()

		if SHOW_DEBUG_WINDOW:
			cv2.imshow('Monitor Distracciones', frame)
			# Si el usuario cierra la ventana, getWindowProperty devuelve 0 -> salir
			try:
				visible = cv2.getWindowProperty('Monitor Distracciones', cv2.WND_PROP_VISIBLE)
			except Exception:
				visible = 1
			if visible < 1:
				print('Ventana cerrada por el usuario, finalizando.')
				stop_event.set()
				break

		key = cv2.waitKey(1) & 0xFF
		if key == ord('q'):
			stop_event.set()
			break

		# Comprobar duración máxima
		if max_duration > 0 and (time.time() - start_time) >= max_duration:
			print('Duración máxima alcanzada, finalizando.')
			stop_event.set()
			break

	# Señalar al worker que pare y esperar join limpio
	stop_event.set()
	try:
		worker.join(timeout=3.0)
	except Exception:
		pass

	cap.release()
	try:
		face.close()
	except Exception:
		pass
	cv2.destroyAllWindows()


if __name__ == '__main__':
	try:
		main()
	except KeyboardInterrupt:
		print('\nInterrumpido por el usuario')


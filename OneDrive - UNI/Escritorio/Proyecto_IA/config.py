# Configuraciones y umbrales del sistema de detección
CAMERA_INDEX = 0  # Índice de la cámara por defecto

# Umbrales para detección de atención/rostro
EYE_AR_THRESHOLD = 0.18  # Umbral para considerar ojos cerrados (EAR)
EYE_AR_CONSEC_FRAMES = 3  # Frames consecutivos para confirmar ojo cerrado
HEAD_TURN_THRESHOLD = 0.12  # Desplazamiento normalizado del nariz para considerar mirada lateral

# Umbral de orientación de cabeza (grados). Se debe permitir cierto ángulo para mirar espejos.
HEAD_YAW_THRESHOLD_DEG = 30.0

# Umbrales para detección de celular (ultralytics)
PHONE_CONF_THRESHOLD = 0.4

# Alertas y UI
ALERT_DURATION_SECONDS = 1.0
ALERT_USE_BEEP = True  # En Windows usa winsound.Beep si es True
ALERT_FREQUENCY = 1000  # Hz para el beep
ALERT_DURATION_MS = 500  # ms para el beep

SHOW_DEBUG_WINDOW = True

# Tiempo de distracción necesario para lanzar alerta
DISTRACTION_SECONDS_TO_ALERT = 2.0

# Ruta por defecto del modelo de detección de objetos (ultralytics)
PHONE_MODEL_PATH = 'yolov8n.pt'

# Carpeta para audios de alerta
AUDIOS_DIR = 'audios'

# Tiempo mínimo entre eventos guardados (segundos) para evitar grabados repetidos
EVENT_COOLDOWN_SECONDS = 10.0

# Detección de teléfono: requerir N detecciones consecutivas y revisar cada K frames
PHONE_DETECT_CONSECUTIVE = 3
PHONE_DETECT_INTERVAL = 3
PHONE_IMG_SIZE = 320

# Parámetros de suavizado temporal y detección por ventana
DETECTION_WINDOW_SECONDS = 2.0  # ventana temporal para confirmar detecciones
PHONE_REQUIRED_IN_WINDOW = 3    # cuántas detecciones de teléfono en la ventana para confirmar
FACE_REQUIRED_IN_WINDOW = 2     # cuántas detecciones faciales (ojos cerrados/head turn) en la ventana

# Guardado/thumbnail
SAVE_VIDEO_FOURCC = 'mp4v'  # fourcc para mp4 (H264/MP4 compatible depende del sistema)
THUMBNAIL_SIZE = (320, 180)

# Privacidad y retención. Por defecto no se guardan imágenes ni vídeo.
SAVE_EVENT_FRAMES = False
EVENT_RETENTION_DAYS = 30
EVENT_MAX_FILES = 500
EVENT_CLEANUP_INTERVAL_SECONDS = 300.0
EVENT_ENCRYPT_METADATA = True
EVENT_ENCRYPTION_KEY_ENV = 'EVENT_ENCRYPTION_KEY'


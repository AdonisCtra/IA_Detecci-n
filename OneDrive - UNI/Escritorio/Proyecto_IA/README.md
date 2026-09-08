Proyecto de monitoreo de distracciones (visión + IA)

Pasos rápidos para ejecutar:

1. Crear y activar un entorno virtual (Windows PowerShell):

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

2. Instalar dependencias:

```powershell
pip install -r requirements.txt
```

3. Ejecutar:

```powershell
python main.py
```

### Privacidad, cifrado y retención

Por defecto el sistema no guarda frames ni vídeo. Registra únicamente el metadato del evento y, si se configura una clave Fernet, lo cifra junto con cualquier artefacto permitido. Genera una clave una vez y guárdala como secreto del entorno:

```powershell
$env:EVENT_ENCRYPTION_KEY = python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
python main.py
```

Para permitir explícitamente vídeo/miniaturas, ejecutar `python main.py --save-frames`. Los archivos se escriben de forma atómica y quedan como `.enc`. La retención se puede ajustar con `--retention-days 30` y `--max-event-files 500`; `--no-encrypt-events` solo debe usarse en desarrollo.

### Calibración guiada

Ejecuta `python main.py --calibrate --calibrate-duration 8`. La ventana indicará que mires al frente, mostrará el progreso y permitirá cancelar con `q`.

4. Descargar modelo YOLO (opcional, el script intentará hacerlo automáticamente):

```powershell
python download_yolo.py
```

Notas:
- Asegúrate de tener la cámara disponible.
- Si quieres detectar celulares con YOLO, coloca el modelo `yolov8n.pt` en la raíz o cambia `PHONE_MODEL_PATH` en `config.py`.
- Los módulos usan `mediapipe` y `ultralytics`. `download_yolo.py` intenta forzar la descarga de `yolov8n.pt` usando `ultralytics` y copiarlo a la raíz. Asegúrate de tener conexión a Internet para la descarga automática.

### Recolección de datos

Usa `collect_dataset.py` para recolectar ejemplos manualmente (presiona `p` para marcar como phone, `n` para not-phone):

```bash
python collect_dataset.py --out dataset --model yolov8n.pt --imgsz 320
```

Los ejemplos se guardarán en `dataset/phone` y `dataset/not_phone`.

### Entrenamiento/verificación

Hay un esqueleto de clasificador secundario en `classifier.py` que puede usarse para entrenar un verificador sobre `dataset/` (ImageFolder layout):

```bash
python classifier.py --data dataset --epochs 5 --out classifier.pth
```

### Exportar a ONNX

Para exportar el modelo YOLO a ONNX (si `ultralytics` está instalado):

```bash
python export_onnx.py --model yolov8n.pt --imgsz 320
```

### GPU

Consulta `gpu_utils.py` para detectar si PyTorch encuentra `cuda`.


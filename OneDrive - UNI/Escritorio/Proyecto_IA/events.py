import os
import time
from pathlib import Path
import logging
import cv2
from typing import Deque, List
import json
from datetime import datetime
from config import (SAVE_VIDEO_FOURCC, THUMBNAIL_SIZE, EVENT_RETENTION_DAYS,
                    EVENT_MAX_FILES, EVENT_ENCRYPT_METADATA,
                    EVENT_ENCRYPTION_KEY_ENV)

try:
    from cryptography.fernet import Fernet
except Exception:
    Fernet = None


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


class EventRecorder:
    def __init__(self, out_dir: str = 'events', log_file: str = 'events/events.log',
                 encrypt_metadata: bool = EVENT_ENCRYPT_METADATA):
        self.out_dir = Path(out_dir)
        ensure_dir(self.out_dir)
        self.log_file = Path(log_file)
        ensure_dir(str(self.log_file.parent))
        logging.basicConfig(filename=str(self.log_file), level=logging.INFO,
                            format='%(asctime)s %(levelname)s %(message)s')
        self.encrypt_metadata = encrypt_metadata
        self._fernet = self._load_fernet()

    def _load_fernet(self):
        if not self.encrypt_metadata:
            return None
        if Fernet is None:
            logging.warning('cryptography no está instalada; se omitirá el guardado cifrado')
            return None
        key = os.getenv(EVENT_ENCRYPTION_KEY_ENV)
        if not key:
            logging.warning('Falta %s; se omitirá el guardado cifrado', EVENT_ENCRYPTION_KEY_ENV)
            return None
        try:
            return Fernet(key.encode('ascii'))
        except Exception:
            logging.error('Clave Fernet inválida en %s; se omitirá el guardado cifrado', EVENT_ENCRYPTION_KEY_ENV)
            return None

    def _write_metadata(self, path: Path, metadata: dict):
        if self.encrypt_metadata and self._fernet is None:
            logging.error('Metadatos omitidos: no hay una clave Fernet válida en %s', EVENT_ENCRYPTION_KEY_ENV)
            return ''
        payload = json.dumps(metadata, ensure_ascii=False, indent=2).encode('utf-8')
        if self._fernet is not None:
            path = path.with_suffix(path.suffix + '.enc')
            payload = self._fernet.encrypt(payload)
        temp_path = path.with_suffix(path.suffix + '.tmp')
        try:
            with open(temp_path, 'wb') as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, path)
        finally:
            if temp_path.exists():
                temp_path.unlink(missing_ok=True)
        return str(path)

    def _protect_file(self, path: Path) -> str:
        if self._fernet is None:
            return str(path)
        encrypted_path = path.with_suffix(path.suffix + '.enc')
        temp_path = encrypted_path.with_suffix(encrypted_path.suffix + '.tmp')
        with open(path, 'rb') as source:
            encrypted = self._fernet.encrypt(source.read())
        try:
            with open(temp_path, 'wb') as target:
                target.write(encrypted)
                target.flush()
                os.fsync(target.fileno())
            os.replace(temp_path, encrypted_path)
            path.unlink(missing_ok=True)
        finally:
            temp_path.unlink(missing_ok=True)
        return str(encrypted_path)

    def cleanup(self, retention_days: int = EVENT_RETENTION_DAYS,
                max_files: int = EVENT_MAX_FILES):
        """Elimina eventos antiguos y conserva como máximo `max_files` grupos."""
        candidates = [p for p in self.out_dir.iterdir() if p.is_file() and p.name.startswith('event_')]
        groups = {}
        for path in candidates:
            groups.setdefault(path.stem.split('.')[0], []).append(path)
        now = time.time()
        expired = []
        for key, group in groups.items():
            newest = max(p.stat().st_mtime for p in group)
            if retention_days >= 0 and now - newest > retention_days * 86400:
                for path in group:
                    path.unlink(missing_ok=True)
                expired.append(key)
        for key in expired:
            groups.pop(key, None)
        ordered = sorted(groups.values(), key=lambda group: max(p.stat().st_mtime for p in group), reverse=True)
        for group in ordered[max_files:]:
            for path in group:
                path.unlink(missing_ok=True)

    def save_metadata(self, label: str = 'distraccion', reasons: List[str] = None,
                      fps: float = 0.0, frame_count: int = 0) -> str:
        base = Path(self._filename(label))
        metadata = {
            'label': label,
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'fps': fps,
            'frame_count': frame_count,
            'reasons': reasons or [],
            'frames_saved': False,
        }
        return self._write_metadata(base.with_suffix('.json'), metadata)

    def _filename(self, label: str) -> str:
        ts = time.strftime('%Y%m%d_%H%M%S')
        return str(self.out_dir / f"event_{label}_{ts}")

    def save_event_from_buffer(self, buffer: Deque, cap: cv2.VideoCapture, post_seconds: float = 3.0, fps: float = 20.0, label: str = 'distraccion', reasons: List[str] = None) -> str:
        """Guarda los frames del buffer y captura `post_seconds` adicionales desde `cap`.
        Devuelve la ruta del archivo guardado o cadena vacía si falla."""
        frames = list(buffer)
        if not frames:
            logging.warning('Buffer vacío, no hay frames para guardar')
            return ''
        if self.encrypt_metadata and self._fernet is None:
            logging.error('Evento omitido: no hay clave de cifrado válida')
            return ''

        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*SAVE_VIDEO_FOURCC)
        base = Path(self._filename(label))
        final_path = base.with_suffix('.mp4')
        out_path = str(base.with_name(base.name + '.tmp.mp4'))
        try:
            writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
        except Exception as e:
            logging.error(f'No se pudo crear VideoWriter: {e}')
            return ''

        # escribir buffer
        for f in frames:
            writer.write(f)

        # capturar y escribir post_frames
        post_frames = int(post_seconds * fps)
        for _ in range(post_frames):
            ret, f = cap.read()
            if not ret:
                break
            writer.write(f)

        writer.release()
        os.replace(out_path, final_path)
        out_path = self._protect_file(final_path)
        # crear thumbnail (primer frame)
        try:
            thumb_file = base.with_suffix('.jpg')
            thumb_tmp = base.with_name(base.name + '.tmp.jpg')
            thumb = cv2.resize(frames[0], THUMBNAIL_SIZE)
            cv2.imwrite(str(thumb_tmp), thumb)
            os.replace(thumb_tmp, thumb_file)
            thumb_path = self._protect_file(thumb_file)
        except Exception:
            thumb_path = ''

        # guardar metadatos JSON
        meta = {
            'path': out_path,
            'label': label,
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'fps': fps,
            'frame_count': len(frames),
            'thumbnail': thumb_path,
            'reasons': reasons or []
        }
        try:
            self._write_metadata(base.with_suffix('.json'), meta)
        except Exception as exc:
            logging.error('No se pudieron guardar metadatos: %s', exc)

        logging.info(f'Evento guardado: {out_path} (label={label})')
        return out_path

    def save_frames(self, frames: List, fps: float = 20.0, label: str = 'distraccion', reasons: List[str] = None) -> str:
        """Guarda una lista de frames (ya copiadas) en un archivo AVI y retorna la ruta."""
        if not frames:
            logging.warning('No hay frames para guardar en save_frames')
            return ''
        if self.encrypt_metadata and self._fernet is None:
            logging.error('Evento omitido: no hay clave de cifrado válida')
            return ''

        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*SAVE_VIDEO_FOURCC)
        base = Path(self._filename(label))
        final_path = base.with_suffix('.mp4')
        out_path = str(base.with_name(base.name + '.tmp.mp4'))
        try:
            writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
        except Exception as e:
            logging.error(f'No se pudo crear VideoWriter en save_frames: {e}')
            return ''

        for f in frames:
            writer.write(f)

        writer.release()
        os.replace(out_path, final_path)
        out_path = self._protect_file(final_path)
        # thumbnail
        try:
            thumb_file = base.with_suffix('.jpg')
            thumb_tmp = base.with_name(base.name + '.tmp.jpg')
            thumb = cv2.resize(frames[0], THUMBNAIL_SIZE)
            cv2.imwrite(str(thumb_tmp), thumb)
            os.replace(thumb_tmp, thumb_file)
            thumb_path = self._protect_file(thumb_file)
        except Exception:
            thumb_path = ''

        meta = {
            'path': out_path,
            'label': label,
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'fps': fps,
            'frame_count': len(frames),
            'thumbnail': thumb_path,
            'reasons': reasons or []
        }
        try:
            self._write_metadata(base.with_suffix('.json'), meta)
        except Exception as exc:
            logging.error('No se pudieron guardar metadatos: %s', exc)

        logging.info(f'Evento guardado (save_frames): {out_path} (label={label})')
        return out_path

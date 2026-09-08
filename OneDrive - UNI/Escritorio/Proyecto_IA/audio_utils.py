import os
import wave
import struct
import math
import sys
from config import AUDIOS_DIR

try:
    import winsound
    _HAS_WINSOUND = True
except Exception:
    winsound = None
    _HAS_WINSOUND = False

try:
    import pygame
    _HAS_PYGAME = True
except Exception:
    pygame = None
    _HAS_PYGAME = False


def ensure_audios_dir():
    if not os.path.exists(AUDIOS_DIR):
        os.makedirs(AUDIOS_DIR, exist_ok=True)


def generate_beep_wav(path: str, freq: int = 1000, duration_ms: int = 500, volume: float = 0.5):
    # Genera un WAV mono 16-bit PCM con una seno de la frecuencia dada
    framerate = 44100
    nframes = int(framerate * (duration_ms / 1000.0))
    amplitude = int(32767 * volume)
    with wave.open(path, 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(framerate)
        for i in range(nframes):
            t = float(i) / framerate
            val = int(amplitude * math.sin(2.0 * math.pi * freq * t))
            data = struct.pack('<h', val)
            wf.writeframesraw(data)
        wf.writeframes(b'')


def ensure_alert_sound(filename: str = 'alert.wav', freq: int = 1000, duration_ms: int = 500):
    ensure_audios_dir()
    path = os.path.join(AUDIOS_DIR, filename)
    if not os.path.exists(path):
        try:
            generate_beep_wav(path, freq=freq, duration_ms=duration_ms)
        except Exception:
            # Si falla la generación, ignorar; winsound podría usarse
            pass
    return path


def play_alert(path: str = None, freq: int = 1000, duration_ms: int = 500):
    if path is None:
        path = ensure_alert_sound(duration_ms=duration_ms, freq=freq)

    # Priorizar winsound en Windows
    if _HAS_WINSOUND and sys.platform.startswith('win'):
        try:
            winsound.PlaySound(path, winsound.SND_FILENAME | winsound.SND_ASYNC)
            return True
        except Exception:
            try:
                winsound.Beep(freq, duration_ms)
                return True
            except Exception:
                pass

    # Intentar con pygame
    if _HAS_PYGAME:
        try:
            pygame.mixer.init()
            pygame.mixer.music.load(path)
            pygame.mixer.music.play()
            return True
        except Exception:
            try:
                # reproducir por winsound si está disponible
                if _HAS_WINSOUND:
                    winsound.Beep(freq, duration_ms)
                    return True
            except Exception:
                pass

    # Fallback: imprimir
    print("ALERTA (no audio disponible): distracción detectada")
    return False

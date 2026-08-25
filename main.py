import io
import sys
import os
# Включаем штатную диагностику OpenCV до первого import cv2. В EXE она
# полезна при обращении к драйверам DirectShow/Media Foundation.
os.environ.setdefault("OPENCV_VIDEOIO_DEBUG", "1")
os.environ.setdefault("OPENCV_LOG_LEVEL", "DEBUG")
import cv2
import numpy as np
import torch
import uuid
import base64
import time
import threading
import subprocess
import queue
import codecs
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk, ImageDraw
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct, PointIdsList
from torchvision.models import resnet50, ResNet50_Weights

# --- ИМПОРТЫ ДЛЯ FLASK ---
from flask import Flask, Response, render_template_string, jsonify, request, send_from_directory

if sys.stdout is not None and hasattr(sys.stdout, 'buffer'):
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
else:
    try:
        sys.stdout = open(os.devnull, "w", encoding='utf-8')
    except:
        pass

if sys.stderr is not None and hasattr(sys.stderr, 'buffer'):
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')
else:
    try:
        sys.stderr = open(os.devnull, "w", encoding='utf-8')
    except:
        pass

# --- НАСТРОЙКИ СИСТЕМЫ ---
if getattr(sys, 'frozen', False):
    # Если запущено как .exe
    BASE_PATH = os.path.dirname(sys.executable)
else:
    # Если запущено как обычный скрипт .py
    BASE_PATH = os.path.dirname(os.path.abspath(__file__))

CAMERA_INDEX_FILE = os.path.join(BASE_PATH, "camera_index.txt")
COLLECTION_NAME = "parts_resnet50"

BASE_DIR = os.path.join(BASE_PATH, "reference_images")
QDRANT_DIR = os.path.join(BASE_PATH, "qdrant_storage")

os.makedirs(BASE_DIR, exist_ok=True)
os.makedirs(QDRANT_DIR, exist_ok=True)
HOST_PORT = 5000
CAMERA_LOG_FILE = os.path.join(BASE_PATH, "camera_debug.log")

def camera_log(message):
    """Пишем диагностику камеры в файл, потому что EXE запускается с --noconsole."""
    text = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    try:
        with open(CAMERA_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(text + "\n")
    except Exception:
        pass


def camera_log_system_diagnostics():
    """Пишет в camera_debug.log максимум данных без изменения настроек ПК."""
    camera_log("========== РАСШИРЕННАЯ ДИАГНОСТИКА КАМЕРЫ ==========")
    camera_log(f"Python: {sys.version.replace(chr(10), ' ')}")
    camera_log(f"EXE: {sys.executable}")
    camera_log(f"OpenCV: {cv2.__version__}; файл: {getattr(cv2, '__file__', 'неизвестно')}")
    camera_log(f"Windows: {os.name}; frozen={getattr(sys, 'frozen', False)}")
    try:
        backend_ids = cv2.videoio_registry.getCameraBackends()
        backend_names = [cv2.videoio_registry.getBackendName(x) for x in backend_ids]
        camera_log(f"Доступные camera backends OpenCV: {list(zip(backend_ids, backend_names))}")
    except Exception as e:
        camera_log(f"Не удалось получить backends OpenCV: {repr(e)}")

    if os.name != "nt":
        return

    commands = {
        "PnP camera/image devices": (
            "Get-PnpDevice -PresentOnly | Where-Object { $_.Class -in "
            "'Camera','Image' } | Format-List Status,Class,FriendlyName,InstanceId"
        ),
        "Camera-related processes": (
            "Get-Process | Where-Object { $_.ProcessName -match "
            "'camera|zoom|teams|skype|discord|obs|browser' } | "
            "Select-Object ProcessName,Id,Path | Format-Table -AutoSize"
        ),
    }
    for title, command in commands.items():
        try:
            result = subprocess.run(
                ["powershell", "-NoProfile", "-Command", command],
                capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=12,
            )
            output = (result.stdout or result.stderr or "нет данных").strip()
            camera_log(f"{title} (exit={result.returncode}):\n{output}")
        except Exception as e:
            camera_log(f"{title}: ошибка запуска диагностики: {repr(e)}")
    try:
        print(text)
    except Exception:
        pass


def load_camera_index():
    if os.path.exists(CAMERA_INDEX_FILE):
        try:
            with open(CAMERA_INDEX_FILE, "r", encoding="utf-8") as f:
                return int(f.read().strip())
        except:
            pass
    return 0

def save_camera_index(idx):
    with open(CAMERA_INDEX_FILE, "w", encoding="utf-8") as f:
        f.write(str(idx))

CAMERA_INDEX = load_camera_index()

# Глобальные переменные состояния
global_camera_active = False
latest_frame = None       # Полный UI-кадр
latest_crop = None        # Кроп 400x400 для нейросети
latest_raw_crop = None    # Чистый кроп для сохранения (без графики)
state_lock = threading.Lock()

# Блокировка для безопасной работы с Qdrant из разных потоков
qdrant_lock = threading.Lock()

scan_results = {
    "part": "Ожидание...", "score": "0%",
    "status_text": "СКАНИРОВАНИЕ...", "status_bg": "#F59E0B",
    "sim1_name": "---", "sim1_score": "0%", "path_sim1": "",
    "sim2_name": "---", "sim2_score": "0%", "path_sim2": "",
    "path_best": ""
}

# --- ИНИЦИАЛИЗАЦИЯ ИИ И ЛОКАЛЬНОГО QDRANT ---
print("⚙️ Инициализация локального Qdrant и ResNet50...")
client = QdrantClient(path=QDRANT_DIR)
if not client.collection_exists(COLLECTION_NAME):
    client.create_collection(
        COLLECTION_NAME, 
        vectors_config=VectorParams(size=2048, distance=Distance.COSINE)
    )
resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
resnet.fc = torch.nn.Identity()
resnet.eval()
preprocess = ResNet50_Weights.DEFAULT.transforms()
print("✅ ИИ и База данных успешно инициализированы!")

def load_list(filename, default):
    filepath = os.path.join(BASE_PATH, filename)
    if not os.path.exists(filepath):
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(default))
        return default
    with open(filepath, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

def save_list(filename, data_list):
    filepath = os.path.join(BASE_PATH, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(data_list))


# =====================================================================
# CAMERA MANAGER
# Один владелец VideoCapture + один reader thread.
# Tkinter и Flask получают только копию последнего кадра.
# =====================================================================
class CameraManager:
    """
    Надежная работа с Windows-камерами для .py и PyInstaller EXE.

    Ключевой принцип:
      - только CameraManager владеет cv2.VideoCapture;
      - только reader thread вызывает cap.read();
      - GUI/Flask никогда не вызывают cap.read()/release();
      - при переключении сначала останавливается reader и освобождается
        старый capture, затем открывается новый;
      - найденный рабочий capture не открывается второй раз.
    """

    MAX_INDEX = 10
    READ_INTERVAL = 0.001
    STARTUP_READS = 12
    STARTUP_DELAY = 0.05
    RECONNECT_DELAY = 0.35

    def __init__(self):
        self.lock = threading.RLock()
        self.frame_lock = threading.Lock()

        self.cap = None
        self.reader_thread = None
        self.stop_event = threading.Event()

        self.index = load_camera_index()
        self.backend_name = ""
        self.backend = None

        self.available_cameras = []
        self.status = "Камера еще не проверялась"
        self.last_error = ""
        self.last_frame = None
        self.last_frame_time = 0.0

        self.frame_failures = 0
        self.dark_frames = 0
        self.running = False
        self.scan_in_progress = False

        self._generation = 0

    # -----------------------------
    # Logging / status
    # -----------------------------
    def _set_status(self, text):
        with self.lock:
            self.status = text

    def get_status(self):
        with self.lock:
            return self.status

    def get_state(self):
        with self.lock:
            active = self.running and self.cap is not None
            return {
                "camera_index": self.index,
                "backend": self.backend_name,
                "available": list(self.available_cameras),
                "active": bool(active),
                "status": self.status,
                "error": self.last_error,
            }

    # -----------------------------
    # Backend selection
    # -----------------------------
    def backend_candidates(self):
        if os.name != "nt":
            return [("ANY", cv2.CAP_ANY)]

        result = []

        msmf = getattr(cv2, "CAP_MSMF", None)
        dshow = getattr(cv2, "CAP_DSHOW", None)
        any_backend = getattr(cv2, "CAP_ANY", 0)

        # MSMF first: it is usually the safer Windows choice for modern UVC cameras.
        if msmf is not None:
            result.append(("MSMF", msmf))
        if dshow is not None:
            result.append(("DSHOW", dshow))
        result.append(("ANY", any_backend))

        # Remove duplicate backend IDs while preserving order.
        seen = set()
        unique = []
        for name, backend in result:
            if backend not in seen:
                seen.add(backend)
                unique.append((name, backend))
        return unique

    def _ordered_backends(self, preferred=None):
        items = self.backend_candidates()
        if preferred:
            items.sort(key=lambda x: 0 if x[0] == preferred else 1)
        return items

    # -----------------------------
    # Frame validation
    # -----------------------------
    @staticmethod
    def frame_info(frame):
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return {
                "valid": False,
                "mean": 0.0,
                "std": 0.0,
                "min": 0,
                "max": 0,
                "nonzero": 0.0,
            }

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean = float(gray.mean())
        std = float(gray.std())
        nonzero = float(np.count_nonzero(gray)) / float(gray.size)

        return {
            "valid": True,
            "mean": mean,
            "std": std,
            "min": int(gray.min()),
            "max": int(gray.max()),
            "nonzero": nonzero,
        }

    @classmethod
    def is_real_frame(cls, frame):
        """
        Не считаем нормальное темное изображение черным.
        Настоящий "black frame" обычно имеет одновременно очень низкие
        mean/std и почти все пиксели одинаковые.
        """
        info = cls.frame_info(frame)
        if not info["valid"]:
            return False

        return not (
            info["mean"] < 3.0
            and info["std"] < 2.0
            and info["nonzero"] < 0.01
        )

    # -----------------------------
    # Open one camera
    # -----------------------------
    def _try_open(self, idx, backend_name, backend):
        cap = None
        try:
            camera_log(f"[CameraManager] Open index={idx}, backend={backend_name}")

            cap = cv2.VideoCapture(idx, backend)

            if cap is None or not cap.isOpened():
                camera_log(
                    f"[CameraManager] index={idx}/{backend_name}: open=FALSE"
                )
                if cap is not None:
                    cap.release()
                return None, None

            try:
                actual = cap.getBackendName()
            except Exception:
                actual = backend_name

            # Do NOT force MJPG or a specific FOURCC here.
            # First let the driver negotiate its default UVC mode.
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass

            # We intentionally do not require a particular resolution.
            # First valid frame wins. This avoids black streams caused by
            # unsupported width/height combinations.
            good_frame = None

            for attempt in range(self.STARTUP_READS):
                started = time.perf_counter()
                try:
                    ret, frame = cap.read()
                except Exception as e:
                    ret, frame = False, None
                    camera_log(
                        f"[CameraManager] index={idx}/{backend_name}: "
                        f"read exception: {repr(e)}"
                    )

                elapsed = (time.perf_counter() - started) * 1000.0

                if ret and frame is not None and frame.size:
                    info = self.frame_info(frame)
                    camera_log(
                        f"[CameraManager] index={idx}/{backend_name}: "
                        f"read {attempt + 1}/{self.STARTUP_READS} OK; "
                        f"shape={frame.shape}; mean={info['mean']:.2f}; "
                        f"std={info['std']:.2f}; min={info['min']}; "
                        f"max={info['max']}; nonzero={info['nonzero']:.3f}; "
                        f"{elapsed:.1f} ms"
                    )

                    if self.is_real_frame(frame):
                        good_frame = frame
                        break
                else:
                    camera_log(
                        f"[CameraManager] index={idx}/{backend_name}: "
                        f"read {attempt + 1}/{self.STARTUP_READS} FALSE; "
                        f"{elapsed:.1f} ms"
                    )

                time.sleep(self.STARTUP_DELAY)

            if good_frame is None:
                camera_log(
                    f"[CameraManager] index={idx}/{backend_name}: "
                    f"opened but no usable image"
                )
                cap.release()
                return None, None

            # Optional autofocus only after a real frame was obtained.
            try:
                cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
            except Exception:
                pass

            try:
                width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                fps = cap.get(cv2.CAP_PROP_FPS)
            except Exception:
                width = height = fps = 0

            camera_log(
                f"[CameraManager] WORKING CAMERA FOUND: "
                f"index={idx}; requested_backend={backend_name}; "
                f"actual_backend={actual}; size={width:.0f}x{height:.0f}; "
                f"fps={fps:.2f}"
            )

            return cap, good_frame

        except Exception as e:
            camera_log(
                f"[CameraManager] open error index={idx}/{backend_name}: {repr(e)}"
            )
            try:
                if cap is not None:
                    cap.release()
            except Exception:
                pass
            return None, None

    def _open_best(self, idx, preferred_backend=None):
        for backend_name, backend in self._ordered_backends(preferred_backend):
            cap, first_frame = self._try_open(idx, backend_name, backend)
            if cap is not None:
                return cap, first_frame, backend_name, backend

        return None, None, "", None

    # -----------------------------
    # Reader thread
    # -----------------------------
    def _reader_loop(self, generation):
        camera_log(
            f"[CameraManager] reader thread START "
            f"generation={generation}, index={self.index}, backend={self.backend_name}"
        )

        local_failures = 0

        while not self.stop_event.is_set():
            with self.lock:
                if generation != self._generation:
                    break
                cap = self.cap

            if cap is None:
                break

            try:
                ret, frame = cap.read()
            except Exception as e:
                camera_log(f"[CameraManager] reader cap.read exception: {repr(e)}")
                ret, frame = False, None

            if ret and frame is not None and frame.size:
                local_failures = 0

                with self.frame_lock:
                    self.last_frame = frame.copy()
                    self.last_frame_time = time.time()

                continue

            local_failures += 1
            with self.lock:
                self.frame_failures = local_failures

            if local_failures in (1, 5, 10, 20, 30):
                camera_log(
                    f"[CameraManager] reader: read failed "
                    f"{local_failures} times; index={self.index}; "
                    f"backend={self.backend_name}"
                )

            # Do not immediately destroy the capture after one failed read.
            time.sleep(0.03)

        camera_log(
            f"[CameraManager] reader thread STOP generation={generation}"
        )

    def _start_reader(self):
        with self.lock:
            self.stop_event.clear()
            self.running = True
            self._generation += 1
            generation = self._generation

            thread = threading.Thread(
                target=self._reader_loop,
                args=(generation,),
                daemon=True,
                name="CameraReader",
            )
            self.reader_thread = thread

        thread.start()

    def _stop_reader_and_release(self):
        with self.lock:
            self._generation += 1
            self.stop_event.set()
            thread = self.reader_thread
            self.reader_thread = None
            cap = self.cap
            self.cap = None
            self.running = False

        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=1.5)

        if cap is not None:
            try:
                cap.release()
            except Exception as e:
                camera_log(f"[CameraManager] release error: {repr(e)}")

        with self.frame_lock:
            self.last_frame = None
            self.last_frame_time = 0.0

    # -----------------------------
    # Connect / switch
    # -----------------------------
    def connect(self, idx=None, preferred_backend=None):
        try:
            idx = self.index if idx is None else int(idx)
        except Exception:
            return False

        camera_log(
            f"[CameraManager] CONNECT requested index={idx}, "
            f"preferred_backend={preferred_backend}"
        )

        # Never let two readers use the same capture.
        self._stop_reader_and_release()

        self._set_status(f"🔄 Подключение камеры #{idx}...")

        cap, first_frame, backend_name, backend = self._open_best(
            idx, preferred_backend=preferred_backend
        )

        if cap is None:
            with self.lock:
                self.last_error = (
                    f"Камера #{idx} не открылась или не дала пригодный кадр"
                )
                self.status = f"❌ Камера #{idx} не дает изображение"
                self.backend_name = ""
                self.backend = None

            camera_log(f"[CameraManager] CONNECT FAILED index={idx}")
            return False

        with self.lock:
            self.cap = cap
            self.index = idx
            self.backend_name = backend_name
            self.backend = backend
            self.frame_failures = 0
            self.dark_frames = 0
            self.last_error = ""
            self.status = (
                f"🟢 Камера #{idx} подключена "
                f"({backend_name})"
            )

        save_camera_index(idx)

        # Seed the shared frame with the known-good startup frame.
        with self.frame_lock:
            self.last_frame = first_frame.copy()
            self.last_frame_time = time.time()

        self._start_reader()

        camera_log(
            f"[CameraManager] CONNECTED index={idx}, backend={backend_name}"
        )
        return True

    def disconnect(self):
        camera_log("[CameraManager] DISCONNECT")
        self._stop_reader_and_release()
        self._set_status("Камера отключена")

    # -----------------------------
    # Discovery
    # -----------------------------
    def scan(self):
        """
        Поиск доступных камер.

        Если рабочая камера уже подключена, ее индекс не открываем второй раз.
        Остальные индексы проверяем временно и сразу освобождаем.
        """
        with self.lock:
            if self.scan_in_progress:
                return list(self.available_cameras)
            self.scan_in_progress = True
            active_index = self.index if self.running and self.cap is not None else None

        try:
            preferred = load_camera_index()
            indexes = [preferred] + [
                i for i in range(self.MAX_INDEX) if i != preferred
            ]

            found = []
            camera_log(
                f"[CameraManager] SCAN START; preferred={preferred}; "
                f"active={active_index}"
            )

            if active_index is not None:
                found.append(active_index)

            for idx in indexes:
                if idx == active_index:
                    continue

                for backend_name, backend in self._ordered_backends():
                    cap, _ = self._try_open(idx, backend_name, backend)
                    if cap is not None:
                        found.append(idx)
                        try:
                            cap.release()
                        except Exception:
                            pass
                        break

            # Stable order and no duplicates.
            found = sorted(set(found))

            with self.lock:
                self.available_cameras = found
                self.status = (
                    f"🟢 Найдены камеры: {', '.join(map(str, found))}"
                    if found else
                    "❌ Камеры не найдены"
                )

            camera_log(f"[CameraManager] SCAN END; found={found}")
            return found

        except Exception as e:
            camera_log(f"[CameraManager] SCAN ERROR: {repr(e)}")
            with self.lock:
                self.last_error = repr(e)
                self.status = "❌ Ошибка поиска камер"
            return []

        finally:
            with self.lock:
                self.scan_in_progress = False

    # -----------------------------
    # Public frame API
    # -----------------------------
    def get_frame(self):
        with self.frame_lock:
            if self.last_frame is None:
                return None
            return self.last_frame.copy()

    def get_frame_age(self):
        with self.frame_lock:
            if self.last_frame_time <= 0:
                return float("inf")
            return max(0.0, time.time() - self.last_frame_time)

    def note_frame_health(self, frame):
        """
        Проверяет уже полученный reader thread кадр.
        Возвращает (is_dark, info).
        """
        info = self.frame_info(frame)
        is_dark = (
            info["valid"]
            and info["mean"] < 3.0
            and info["std"] < 2.0
            and info["nonzero"] < 0.01
        )

        with self.lock:
            if is_dark:
                self.dark_frames += 1
            else:
                self.dark_frames = 0

        return is_dark, info

    def needs_reconnect(self, max_dark_frames=60, max_age=3.0):
        with self.lock:
            dark = self.dark_frames
            failures = self.frame_failures
            running = self.running

        age = self.get_frame_age()

        if not running:
            return True

        if dark >= max_dark_frames:
            return True

        if failures >= 30:
            return True

        if age > max_age:
            return True

        return False

    def diagnostics(self):
        camera_log("========== CAMERA MANAGER DIAGNOSTICS ==========")
        camera_log(f"Python: {sys.version.replace(chr(10), ' ')}")
        camera_log(f"EXE: {sys.executable}")
        camera_log(
            f"OpenCV: {cv2.__version__}; "
            f"file={getattr(cv2, '__file__', 'unknown')}"
        )
        camera_log(
            f"Windows={os.name}; frozen={getattr(sys, 'frozen', False)}"
        )

        try:
            ids = cv2.videoio_registry.getCameraBackends()
            names = []
            for x in ids:
                try:
                    names.append(cv2.videoio_registry.getBackendName(x))
                except Exception:
                    names.append(str(x))
            camera_log(f"OpenCV camera backends: {list(zip(ids, names))}")
        except Exception as e:
            camera_log(f"Cannot enumerate OpenCV backends: {repr(e)}")

        if os.name == "nt":
            commands = {
                "PnP camera/image devices": (
                    "Get-PnpDevice -PresentOnly | "
                    "Where-Object { $_.Class -in 'Camera','Image' } | "
                    "Format-List Status,Class,FriendlyName,InstanceId"
                ),
                "Camera-related processes": (
                    "Get-Process | "
                    "Where-Object { $_.ProcessName -match "
                    "'camera|zoom|teams|skype|discord|obs|browser' } | "
                    "Select-Object ProcessName,Id,Path | "
                    "Format-Table -AutoSize"
                ),
            }

            for title, command in commands.items():
                try:
                    result = subprocess.run(
                        [
                            "powershell",
                            "-NoProfile",
                            "-Command",
                            command,
                        ],
                        capture_output=True,
                        text=True,
                        encoding="utf-8",
                        errors="replace",
                        timeout=12,
                    )
                    output = (
                        result.stdout or result.stderr or "нет данных"
                    ).strip()
                    camera_log(
                        f"{title} (exit={result.returncode}):\n{output}"
                    )
                except Exception as e:
                    camera_log(
                        f"{title}: diagnostic error: {repr(e)}"
                    )

    def shutdown(self):
        camera_log("[CameraManager] SHUTDOWN")
        self._stop_reader_and_release()


# =====================================================================
# TKINTER DESKTOP APP (Сканер + Регистрация + Галерея + Справочники)
# =====================================================================
class IndustrialVisionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Индустриальный Комплекс (Desktop + Web)")
        self.root.geometry("1400x800")
        self.root.configure(bg="#2b2b2b")

        self.ui_queue = queue.Queue()
        self.process_ui_queue()

        # Новый CameraManager: только он владеет VideoCapture.
        self.camera_manager = CameraManager()
        self.camera_lock = self.camera_manager.lock
        self.available_cameras = []
        self.camera_error = "Камера еще не проверялась"
        self.camera_scan_in_progress = False
        self.camera_retry_after = 0
        self.camera_reconnect_scheduled = False
        self.camera_frame_failures = 0
        self.camera_dark_frames = 0
        self.active_camera_backend = ""
        self.system_diagnostics_logged = False

        self.frame_counter = 0
        self.scan_line_y = 0
        self.scan_line_dir = 1
        self.is_inferencing = False
        self.is_detected = False
        
        self.delete_all_confirmed = False
        self.delete_photo_confirmed = False

        self.candidate_name = None
        self.confirm_counter = 0
        self.CONFIRM_FRAMES_FAST = 2
        self.CONFIRM_FRAMES_SLOW = 10
        self.MIN_SCORE = 0.50
        self.patience_counter = 0
        self.MAX_PATIENCE = 3
        self.last_spoken_name = None
        self.audio_lock = threading.Lock()

        style = ttk.Style()
        style.theme_use('default')
        style.configure("TNotebook", background="#2b2b2b", borderwidth=0)
        style.configure("TNotebook.Tab", background="#374151", foreground="white", padding=[20, 10], font=('Arial', 12, 'bold'))
        style.map("TNotebook.Tab", background=[("selected", "#D32F2F")])

        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(expand=True, fill=tk.BOTH)

        self.tab_scanner = tk.Frame(self.notebook, bg="#2b2b2b")
        self.tab_register = tk.Frame(self.notebook, bg="#2b2b2b")
        self.tab_gallery = tk.Frame(self.notebook, bg="#E5E7EB")
        self.tab_metadata = tk.Frame(self.notebook, bg="#2b2b2b")

        self.notebook.add(self.tab_scanner, text="📹 Сканер")
        self.notebook.add(self.tab_register, text="📸 Регистрация")
        self.notebook.add(self.tab_gallery, text="🗂️ Управление базой")
        self.notebook.add(self.tab_metadata, text="⚙️ Справочники")

        self.blank_image = self.create_blank_image((300, 300), "Нет фото")
        self.blank_sim = self.create_blank_image((200, 200), "Пусто")

        self.setup_scanner_ui(self.tab_scanner)
        self.setup_register_ui(self.tab_register)
        self.setup_gallery_ui(self.tab_gallery)
        self.setup_metadata_ui(self.tab_metadata)
        
        self.notebook.bind("<<NotebookTabChanged>>", self.on_tab_changed)

        # Камеру ищем ПОСЛЕ создания интерфейса, чтобы EXE не зависал на старте.
        self.root.after(100, self.start_camera_scan)
        self.root.after(100, self.update_frame)

    # =================================================================
    # КАМЕРА — CameraManager
    # =================================================================
    def set_camera_status(self, text):
        self.camera_error = text
        try:
            if hasattr(self, "camera_status_label"):
                self.camera_status_label.config(text=text)
        except Exception:
            pass

    def open_windows_camera_settings(self):
        try:
            if os.name == "nt":
                os.startfile("ms-settings:privacy-webcam")
                camera_log("Открыты настройки Windows: ms-settings:privacy-webcam")
                return True
        except Exception as e:
            camera_log(f"Не удалось открыть настройки приватности камеры: {e}")
        return False

    def start_camera_scan(self):
        """
        Асинхронный поиск.
        Важно: если камера уже работает, ее VideoCapture не трогаем.
        """
        if self.camera_scan_in_progress:
            return

        self.camera_scan_in_progress = True
        self.set_camera_status("🔎 Ищем доступные камеры...")

        def worker():
            try:
                if not self.system_diagnostics_logged:
                    self.system_diagnostics_logged = True
                    self.camera_manager.diagnostics()

                found = self.camera_manager.scan()
                self.ui_queue.put(
                    lambda found=found: self.finish_camera_scan(found)
                )
            except Exception as e:
                camera_log(
                    f"Критическая ошибка поиска камер: {repr(e)}"
                )
                self.ui_queue.put(
                    lambda: self.finish_camera_scan([])
                )

        threading.Thread(
            target=worker,
            daemon=True,
            name="CameraScanner",
        ).start()

    def finish_camera_scan(self, found):
        self.camera_scan_in_progress = False
        self.available_cameras = list(found)
        self.camera_manager.available_cameras = list(found)

        try:
            if hasattr(self, "cb_camera_idx"):
                values = [str(x) for x in found] if found else [
                    str(load_camera_index())
                ]
                self.cb_camera_idx["values"] = values
                current = str(load_camera_index())
                self.cb_camera_idx.set(
                    current if current in values else values[0]
                )
        except Exception:
            pass

        if not found:
            self.set_camera_status(
                "❌ Камера не найдена. Проверьте USB и разрешения Windows."
            )
            camera_log(
                "Камера не найдена. Открываем настройки приватности Windows."
            )
            try:
                self.open_windows_camera_settings()
                messagebox.showwarning(
                    "Камера не найдена",
                    "Программа не получила рабочий кадр ни с одной камеры.\n\n"
                    "Проверьте:\n"
                    "• Доступ к камере\n"
                    "• Разрешить приложениям доступ к камере\n"
                    "• Разрешить классическим приложениям доступ к камере\n"
                    "• USB-подключение и драйвер камеры\n\n"
                    "После этого нажмите «Найти камеры».\n\n"
                    "Подробный лог: camera_debug.log",
                )
            except Exception:
                pass

            self.root.after(3000, self.start_camera_scan)
            return

        preferred = load_camera_index()
        selected = preferred if preferred in found else found[0]

        # Если активная камера уже выбрана, не открываем ее повторно.
        state = self.camera_manager.get_state()
        if state["active"] and state["camera_index"] == selected:
            self.active_camera_backend = state["backend"]
            self.set_camera_status(
                f"🟢 Камера #{selected} уже работает "
                f"({state['backend']}). "
                f"Доступные: {', '.join(map(str, found))}"
            )
            return

        if self.switch_camera(selected, show_message=False):
            self.set_camera_status(
                f"🟢 Камера #{selected} подключена. "
                f"Доступные: {', '.join(map(str, found))}"
            )
        else:
            self.set_camera_status(
                f"❌ Не удалось подключить камеру #{selected}"
            )

    def scan_available_cameras(self):
        return self.camera_manager.scan()

    def switch_camera(self, idx, show_message=True):
        try:
            idx = int(idx)
        except Exception:
            return False

        self.set_camera_status(f"🔄 Подключение камеры #{idx}...")
        camera_log(f"Переключение на камеру #{idx}")

        state = self.camera_manager.get_state()
        if state["active"] and state["camera_index"] == idx:
            self.active_camera_backend = state["backend"]
            self.set_camera_status(
                f"🟢 Камера #{idx} уже подключена ({state['backend']})"
            )
            return True

        ok = self.camera_manager.connect(idx)

        if not ok:
            self.set_camera_status(
                f"❌ Камера #{idx} не дает изображение"
            )
            if show_message:
                messagebox.showerror(
                    "Ошибка камеры",
                    f"Камера #{idx} не отдает рабочий видеопоток.\n\n"
                    "Попробуйте другую камеру или проверьте разрешения Windows.\n\n"
                    "Смотрите camera_debug.log.",
                )
            return False

        state = self.camera_manager.get_state()
        self.active_camera_backend = state["backend"]
        self.available_cameras = state["available"]
        self.camera_frame_failures = 0
        self.camera_dark_frames = 0
        self.set_camera_status(
            f"🟢 Камера #{idx} подключена ({state['backend']})"
        )
        camera_log(
            f"Рабочая камера: #{idx}, backend={state['backend']}"
        )

        if show_message:
            messagebox.showinfo(
                "Камера",
                f"Камера #{idx} успешно подключена.\n"
                f"Backend: {state['backend']}",
            )

        return True

    def init_camera(self):
        self.start_camera_scan()

    def reconnect_camera_with_next_backend(self, idx):
        """
        Совместимый алиас. CameraManager сам выбирает backend заново.
        """
        try:
            current_backend = self.camera_manager.get_state()["backend"]
            backends = [x[0] for x in self.camera_manager.backend_candidates()]
            preferred = None

            if current_backend in backends:
                pos = backends.index(current_backend)
                if len(backends) > 1:
                    preferred = backends[(pos + 1) % len(backends)]

            self.camera_manager.connect(
                int(idx),
                preferred_backend=preferred,
            )

            state = self.camera_manager.get_state()
            self.active_camera_backend = state["backend"]
            self.set_camera_status(
                f"🟢 Камера #{idx} восстановлена ({state['backend']})"
                if state["active"]
                else f"❌ Не удалось восстановить камеру #{idx}"
            )
        finally:
            self.camera_reconnect_scheduled = False

    def process_ui_queue(self):
        try:
            while True:
                task = self.ui_queue.get_nowait()
                task()
        except queue.Empty:
            pass
        self.root.after(50, self.process_ui_queue)

    def create_blank_image(self, size, text):
        img = Image.new('RGB', size, color='#4B5563')
        draw = ImageDraw.Draw(img)
        draw.text((size[0]//2 - 30, size[1]//2 - 10), text, fill=(200, 200, 200))
        return ImageTk.PhotoImage(img)

    def load_ref_image(self, path, size):
        if path and os.path.exists(path):
            img = Image.open(path)
            img = img.resize(size, Image.Resampling.LANCZOS)
            return ImageTk.PhotoImage(img)
        return self.create_blank_image(size, "Нет файла")

    def setup_scanner_ui(self, parent):
        left_panel = tk.Frame(parent, width=350, bg="#E5E7EB", padx=20, pady=20)
        left_panel.pack(side=tk.LEFT, fill=tk.Y)
        
        tk.Label(left_panel, text="🔍 РЕЗУЛЬТАТ", font=("Arial", 14, "bold"), bg="#E5E7EB", fg="#333").pack(pady=(0, 10))
        self.lbl_best_img = tk.Label(left_panel, image=self.blank_image, bg="#E5E7EB")
        self.lbl_best_img.pack(pady=10)
        self.lbl_part = tk.Label(left_panel, text="Деталь: ...", font=("Arial", 16, "bold"), bg="#E5E7EB", fg="#1D4ED8", wraplength=300)
        self.lbl_part.pack(anchor=tk.W, pady=10)
        self.lbl_score = tk.Label(left_panel, text="Точность: 0%", font=("Arial", 14, "bold"), bg="#E5E7EB", fg="#047857")
        self.lbl_score.pack(anchor=tk.W, pady=10)
        self.status_box = tk.Label(left_panel, text="СКАНИРОВАНИЕ...", font=("Arial", 14, "bold"), bg="#F59E0B", fg="white", pady=10)
        self.status_box.pack(fill=tk.X, side=tk.BOTTOM, pady=20)

        self.video_label_scanner = tk.Label(parent, bg="#2b2b2b")
        self.video_label_scanner.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        right_panel = tk.Frame(parent, width=300, bg="#374151", padx=15, pady=20)
        right_panel.pack(side=tk.RIGHT, fill=tk.Y)
        tk.Label(right_panel, text="БЛИЖАЙШИЕ СОВПАДЕНИЯ", font=("Arial", 12, "bold"), bg="#374151", fg="#D1D5DB").pack(pady=(0, 20))
        
        self.lbl_sim1_img = tk.Label(right_panel, image=self.blank_sim, bg="#374151")
        self.lbl_sim1_img.pack()
        self.lbl_sim1_name = tk.Label(right_panel, text="---", font=("Arial", 11, "bold"), bg="#374151", fg="#9CA3AF")
        self.lbl_sim1_name.pack(pady=2)
        
        self.lbl_sim2_img = tk.Label(right_panel, image=self.blank_sim, bg="#374151")
        self.lbl_sim2_img.pack(pady=(15, 0))
        self.lbl_sim2_name = tk.Label(right_panel, text="---", font=("Arial", 11, "bold"), bg="#374151", fg="#9CA3AF")
        self.lbl_sim2_name.pack(pady=2)

    def setup_register_ui(self, parent):
        self.video_label_register = tk.Label(parent, bg="#2b2b2b")
        self.video_label_register.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        right_panel = tk.Frame(parent, width=400, bg="#E5E7EB", padx=30, pady=30)
        right_panel.pack(side=tk.RIGHT, fill=tk.Y)
        
        tk.Label(right_panel, text="⚙️ РЕГИСТРАЦИЯ ЭТАЛОНА", font=("Arial", 16, "bold"), bg="#E5E7EB", fg="#111827").pack(pady=(0, 20))

        font_label = ("Arial", 12, "bold")
        font_entry = ("Arial", 14)

        tk.Label(right_panel, text="Тип сборки:", bg="#E5E7EB", fg="#4B5563", font=font_label).pack(anchor=tk.W)
        self.cb_type = ttk.Combobox(right_panel, font=font_entry)
        self.cb_type.pack(fill=tk.X, pady=(5, 15))

        tk.Label(right_panel, text="Название детали:", bg="#E5E7EB", fg="#4B5563", font=font_label).pack(anchor=tk.W)
        self.cb_part = ttk.Combobox(right_panel, font=font_entry)
        self.cb_part.pack(fill=tk.X, pady=(5, 25))

        btn_save = tk.Button(right_panel, text="📸 СОХРАНИТЬ РАКУРС", bg="#D32F2F", fg="white", font=("Arial", 14, "bold"), relief=tk.FLAT, command=self.save_reference_from_ui)
        btn_save.pack(fill=tk.X, ipady=10)
        
        self.lbl_reg_status = tk.Label(right_panel, text="", bg="#E5E7EB", font=("Arial", 12, "bold"))
        self.lbl_reg_status.pack(pady=15)

    def load_registration_lists(self):
        types = load_list("types.txt", ["metiz", "bigdetail"])
        self.cb_type['values'] = types
        if types: self.cb_type.set(types[0])
        self.update_part_list()

    def update_part_list(self, *args):
        v_type = self.cb_type.get()
        if not v_type: return
        parts = load_list(f"parts_{v_type}.txt", [f"Деталь_{v_type}"])
        self.cb_part['values'] = parts
        if parts: self.cb_part.set(parts[0])

    def save_reference_from_ui(self):
        global latest_raw_crop, latest_crop
        v_type, part = self.cb_type.get(), self.cb_part.get()
        
        if not all([v_type, part]):
            self.lbl_reg_status.config(text="❌ Заполните все поля!", fg="#EF4444")
            return
            
        with state_lock:
            local_crop = latest_crop.copy() if latest_crop is not None else None
            local_raw = latest_raw_crop.copy() if latest_raw_crop is not None else None

        if local_raw is None or local_crop is None:
            self.lbl_reg_status.config(text="❌ Ошибка камеры!", fg="#EF4444")
            return

        types = load_list("types.txt", [])
        if v_type not in types: save_list("types.txt", types + [v_type])
        parts = load_list(f"parts_{v_type}.txt", [])
        if part not in parts: save_list(f"parts_{v_type}.txt", parts + [part])

        try:
            img_pil = Image.fromarray(cv2.cvtColor(local_crop, cv2.COLOR_BGR2RGB))
            with torch.no_grad():
                vector = resnet(preprocess(img_pil).unsqueeze(0))[0].detach().numpy().tolist()

            point_id = str(uuid.uuid4())
            save_dir = os.path.join(BASE_DIR, v_type, part)
            os.makedirs(save_dir, exist_ok=True)
            cv2.imwrite(os.path.join(save_dir, f"{point_id}.jpg"), local_raw)

            with qdrant_lock:
                client.upsert(
                    COLLECTION_NAME, 
                    [PointStruct(id=point_id, vector=vector, payload={"name": part, "type": v_type, "group_id": f"{v_type}_{part}"})]
                )
            
            self.lbl_reg_status.config(text="✅ Эталон успешно сохранен!", fg="#10B981")
            self.root.after(3000, lambda: self.lbl_reg_status.config(text=""))
        except Exception as e:
            self.lbl_reg_status.config(text=f"❌ Ошибка: {e}", fg="#EF4444")

    def setup_gallery_ui(self, parent):
        top_frame = tk.Frame(parent, bg="#F9FAFB", pady=15, padx=20)
        top_frame.pack(fill=tk.X)
        tk.Button(top_frame, text="🔄 Обновить список", command=self.load_gallery, font=("Arial", 12), bg="#374151", fg="white").pack(side=tk.LEFT)
        
        columns = ("type", "part", "count")
        self.tree = ttk.Treeview(parent, columns=columns, show="headings", height=18)
        self.tree.heading("type", text="Тип сборки")
        self.tree.heading("part", text="Деталь")
        self.tree.heading("count", text="Кол-во эталонов")
        self.tree.column("count", width=150, anchor=tk.CENTER)
        self.tree.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        btn_frame = tk.Frame(parent, bg="#E5E7EB", pady=10)
        btn_frame.pack(fill=tk.X, padx=20)
        
        tk.Button(btn_frame, text="🖼️ Просмотреть фото", command=self.view_part_photos, font=("Arial", 12, "bold"), bg="#1D4ED8", fg="white").pack(side=tk.LEFT, padx=10)
        
        self.btn_del_all = tk.Button(btn_frame, text="🗑️ Удалить всю деталь", command=self.delete_selected_image, font=("Arial", 12, "bold"), bg="#D32F2F", fg="white")
        self.btn_del_all.pack(side=tk.RIGHT)
        
        self.lbl_gallery_status = tk.Label(parent, text="", bg="#E5E7EB", font=("Arial", 11, "bold"))
        self.lbl_gallery_status.pack(pady=5)

    def load_gallery(self):
        self.delete_all_confirmed = False
        self.btn_del_all.config(text="🗑️ Удалить всю деталь", bg="#D32F2F")
        self.lbl_gallery_status.config(text="")
        
        if self.tree.selection():
            self.tree.selection_remove(self.tree.selection())
            
        for i in self.tree.get_children():
            self.tree.delete(i)
            
        if not os.path.exists(BASE_DIR): return
        
        for v_type in os.listdir(BASE_DIR):
            t_path = os.path.join(BASE_DIR, v_type)
            if not os.path.isdir(t_path): continue
            for part in os.listdir(t_path):
                p_path = os.path.join(t_path, part)
                if not os.path.isdir(p_path): continue
                
                photos = [f for f in os.listdir(p_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                if photos:
                    self.tree.insert("", "end", values=(v_type, part, len(photos)))

    def delete_selected_image(self):
        selected = self.tree.selection()
        if not selected:
            self.lbl_gallery_status.config(text="⚠️ Сначала выберите деталь в списке!", fg="#D97706")
            return

        if not self.delete_all_confirmed:
            self.delete_all_confirmed = True
            self.btn_del_all.config(text="⚠️ НАЖМИТЕ ЕЩЕ РАЗ ДЛЯ ПОДТВЕРЖДЕНИЯ", bg="#B71C1C")
            self.lbl_gallery_status.config(text="⚠️ Внимание: будут удалены все фото этой детали навсегда!", fg="#D97706")
            return

        items = [self.tree.item(item, "values") for item in selected]
        self.tree.selection_remove(selected)
        self.btn_del_all.config(text="🗑️ Удаление...", bg="#4B5563")
        self.btn_del_all.config(state=tk.DISABLED)

        def _delete_process(items_to_delete):
            deleted_count = 0
            errors = []

            for item_values in items_to_delete:
                try:
                    v_type, part, _ = item_values
                    p_path = os.path.join(BASE_DIR, v_type, part)
                    if not os.path.exists(p_path): continue

                    files = [f for f in os.listdir(p_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

                    for f in files:
                        point_id = os.path.splitext(f)[0]
                        try:
                            with qdrant_lock:
                                client.delete(
                                    collection_name=COLLECTION_NAME,
                                    points_selector=PointIdsList(points=[point_id]),
                                    wait=True
                                )
                        except Exception as e:
                            errors.append(f"Qdrant: {e}")

                    for f in files:
                        file_path = os.path.join(p_path, f)
                        try:
                            if os.path.exists(file_path):
                                os.remove(file_path)
                                deleted_count += 1
                        except Exception as e:
                            errors.append(f"Файл: {e}")

                    try:
                        if os.path.exists(p_path):
                            os.rmdir(p_path)
                    except: pass
                except Exception as e:
                    errors.append(str(e))

            def _finish_delete():
                self.load_gallery()
                self.btn_del_all.config(state=tk.NORMAL)
                if errors:
                    self.lbl_gallery_status.config(text=f"⚠️ Удалено файлов: {deleted_count} (были ошибки)", fg="#D97706")
                else:
                    self.lbl_gallery_status.config(text=f"✅ Успешно удалено эталонов: {deleted_count}", fg="#10B981")

            self.ui_queue.put(_finish_delete)

        threading.Thread(target=_delete_process, args=(items,), daemon=True).start()

    def view_part_photos(self):
        selected = self.tree.selection()
        if not selected:
            self.lbl_gallery_status.config(text="⚠️ Выберите деталь для просмотра фото!", fg="#D97706")
            return
            
        v_type, part, count = self.tree.item(selected[0], "values")
        p_path = os.path.join(BASE_DIR, v_type, part)
        
        if not os.path.exists(p_path): return
        photos = [f for f in os.listdir(p_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if not photos: return
        
        viewer = tk.Toplevel(self.root)
        viewer.title(f"Галерея ракурсов: {part} ({v_type})")
        viewer.geometry("700x700")
        viewer.configure(bg="#2b2b2b")
        viewer.grab_set() 
        
        viewer.current_idx = 0
        viewer.photos_list = photos
        viewer.delete_confirmed = False
        
        lbl_info = tk.Label(viewer, text="", font=("Arial", 14, "bold"), bg="#2b2b2b", fg="white")
        lbl_info.pack(pady=10)
        
        lbl_img = tk.Label(viewer, bg="#2b2b2b")
        lbl_img.pack(expand=True)
        
        lbl_viewer_status = tk.Label(viewer, text="", bg="#2b2b2b", font=("Arial", 11, "bold"))
        lbl_viewer_status.pack(pady=5)
        
        btn_frame = tk.Frame(viewer, bg="#2b2b2b")
        btn_frame.pack(fill=tk.X, pady=15, padx=30)
        
        def render_photo():
            viewer.delete_confirmed = False
            btn_del_photo.config(text="🗑️ Удалить это фото", bg="#D32F2F")
            lbl_viewer_status.config(text="")
            
            if not viewer.photos_list:
                viewer.destroy()
                self.load_gallery()
                return
            
            viewer.current_idx = viewer.current_idx % len(viewer.photos_list)
            current_file = viewer.photos_list[viewer.current_idx]
            lbl_info.config(text=f"Фото {viewer.current_idx + 1} из {len(viewer.photos_list)}\nID: {os.path.splitext(current_file)[0]}")
            
            img_path = os.path.join(p_path, current_file)
            img = Image.open(img_path)
            img.thumbnail((450, 450), Image.Resampling.LANCZOS)
            viewer.img_tk = ImageTk.PhotoImage(img)
            lbl_img.config(image=viewer.img_tk)

        def next_photo(): viewer.current_idx += 1; render_photo()
        def prev_photo(): viewer.current_idx -= 1; render_photo()

        def delete_current_photo():
            if not viewer.photos_list: return
            
            if not viewer.delete_confirmed:
                viewer.delete_confirmed = True
                btn_del_photo.config(text="⚠️ НАЖМИТЕ ЕЩЕ РАЗ ДЛЯ УДАЛЕНИЯ", bg="#B71C1C")
                lbl_viewer_status.config(text="⚠️ Нажмите кнопку еще раз, чтобы удалить этот ракурс", fg="#D97706")
                return

            current_file = viewer.photos_list[viewer.current_idx]
            point_id = os.path.splitext(current_file)[0]
            img_path = os.path.join(p_path, current_file)
            
            def _del_single():
                try:
                    with qdrant_lock:
                        client.delete(
                            collection_name=COLLECTION_NAME, 
                            points_selector=PointIdsList(points=[point_id]),
                            wait=True
                        )
                except: pass
                
                try: os.remove(img_path)
                except: pass
                
                def _update_ui():
                    if viewer.photos_list:
                        viewer.photos_list.pop(viewer.current_idx)
                    render_photo()
                    self.load_gallery()
                
                self.ui_queue.put(_update_ui)
                
            threading.Thread(target=_del_single, daemon=True).start()

        tk.Button(btn_frame, text="⬅️ Назад", command=prev_photo, font=("Arial", 12), bg="#374151", fg="white", width=10).pack(side=tk.LEFT)
        btn_del_photo = tk.Button(btn_frame, text="🗑️ Удалить это фото", command=delete_current_photo, font=("Arial", 11, "bold"), bg="#D32F2F", fg="white")
        btn_del_photo.pack(side=tk.LEFT, expand=True, padx=10)
        tk.Button(btn_frame, text="Вперед ➡️", command=next_photo, font=("Arial", 12), bg="#374151", fg="white", width=10).pack(side=tk.RIGHT)
        
        render_photo()

    def setup_metadata_ui(self, parent):
        main_frame = tk.Frame(parent, bg="#2b2b2b", padx=20, pady=20)
        main_frame.pack(fill=tk.BOTH, expand=True)

        tk.Label(main_frame, text="⚙️ Управление справочниками и настройками", font=("Arial", 16, "bold"), bg="#2b2b2b", fg="white").pack(pady=(0, 10))

        cam_frame = tk.Frame(main_frame, bg="#374151", padx=15, pady=10)
        cam_frame.pack(fill=tk.X, pady=(0, 15))
        tk.Label(cam_frame, text="📹 Камера:", font=("Arial", 11, "bold"), bg="#374151", fg="white").pack(side=tk.LEFT, padx=10)
        self.cb_camera_idx = ttk.Combobox(cam_frame, values=[str(load_camera_index())], font=("Arial", 11), width=8, state="readonly")
        self.cb_camera_idx.pack(side=tk.LEFT, padx=10)
        self.cb_camera_idx.set(str(load_camera_index()))
        tk.Button(cam_frame, text="🔎 Найти камеры", bg="#3B82F6", fg="white", font=("Arial", 10, "bold"), command=self.start_camera_scan).pack(side=tk.LEFT, padx=5)
        tk.Button(cam_frame, text="🔌 Подключить", bg="#10B981", fg="white", font=("Arial", 10, "bold"), command=self.apply_camera_index).pack(side=tk.LEFT, padx=5)
        tk.Button(cam_frame, text="🔐 Разрешения Windows", bg="#F59E0B", fg="white", font=("Arial", 10, "bold"), command=self.open_windows_camera_settings).pack(side=tk.LEFT, padx=5)
        self.camera_status_label = tk.Label(cam_frame, text="Проверка...", font=("Arial", 10, "bold"), bg="#374151", fg="#D1D5DB")
        self.camera_status_label.pack(side=tk.LEFT, padx=15)

        container = tk.Frame(main_frame, bg="#2b2b2b")
        container.pack(fill=tk.BOTH, expand=True)
        container.columnconfigure(0, weight=1)
        container.columnconfigure(1, weight=1)
        container.rowconfigure(0, weight=1)

        font_label = ("Arial", 11, "bold")
        font_btn = ("Arial", 10, "bold")

        f_type = tk.LabelFrame(container, text=" Типы сборки (Types) ", font=font_label, bg="#374151", fg="white", padx=15, pady=15)
        f_type.grid(row=0, column=0, sticky="nsew", padx=10)
        
        self.lb_types = tk.Listbox(f_type, font=("Arial", 12), height=10)
        self.lb_types.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        self.ent_type = tk.Entry(f_type, font=("Arial", 12))
        self.ent_type.pack(fill=tk.X, pady=(0, 10))
        
        b_frame2 = tk.Frame(f_type, bg="#374151")
        b_frame2.pack(fill=tk.X)
        tk.Button(b_frame2, text="➕ Добавить", bg="#10B981", fg="white", font=font_btn, command=self.meta_add_type).pack(side=tk.LEFT, expand=True, padx=2)
        tk.Button(b_frame2, text="✏️ Изменить", bg="#3B82F6", fg="white", font=font_btn, command=self.meta_edit_type).pack(side=tk.LEFT, expand=True, padx=2)
        tk.Button(b_frame2, text="🗑️ Удалить", bg="#D32F2F", fg="white", font=font_btn, command=self.meta_delete_type).pack(side=tk.LEFT, expand=True, padx=2)
        self.lb_types.bind("<<ListboxSelect>>", lambda e: self.meta_select_item(self.lb_types, self.ent_type))

        f_part = tk.LabelFrame(container, text=" Детали (Parts) ", font=font_label, bg="#374151", fg="white", padx=15, pady=15)
        f_part.grid(row=0, column=1, sticky="nsew", padx=10)

        tk.Label(f_part, text="Выберите тип для деталей:", bg="#374151", fg="#D1D5DB", font=("Arial", 10)).pack(anchor=tk.W)
        self.cb_meta_type = ttk.Combobox(f_part, font=("Arial", 11), state="readonly")
        self.cb_meta_type.pack(fill=tk.X, pady=(2, 10))
        self.cb_meta_type.bind("<<ComboboxSelected>>", self.load_meta_parts)
        
        self.lb_parts = tk.Listbox(f_part, font=("Arial", 12), height=8)
        self.lb_parts.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        self.ent_part = tk.Entry(f_part, font=("Arial", 12))
        self.ent_part.pack(fill=tk.X, pady=(0, 10))
        
        b_frame3 = tk.Frame(f_part, bg="#374151")
        b_frame3.pack(fill=tk.X)
        tk.Button(b_frame3, text="➕ Добавить", bg="#10B981", fg="white", font=font_btn, command=self.meta_add_part).pack(side=tk.LEFT, expand=True, padx=2)
        tk.Button(b_frame3, text="✏️ Изменить", bg="#3B82F6", fg="white", font=font_btn, command=self.meta_edit_part).pack(side=tk.LEFT, expand=True, padx=2)
        tk.Button(b_frame3, text="🗑️ Удалить", bg="#D32F2F", fg="white", font=font_btn, command=self.meta_delete_part).pack(side=tk.LEFT, expand=True, padx=2)
        self.lb_parts.bind("<<ListboxSelect>>", lambda e: self.meta_select_item(self.lb_parts, self.ent_part))

    def apply_camera_index(self):
        try:
            new_idx = int(self.cb_camera_idx.get())
            if self.switch_camera(new_idx, show_message=True):
                self.cb_camera_idx.set(str(new_idx))
        except Exception as e:
            camera_log(f"Ошибка переключения камеры из GUI: {repr(e)}")
            messagebox.showerror("Камера", str(e))

    def load_metadata_tab(self):
        types = load_list("types.txt", ["metiz", "bigdetail"])
        self.lb_types.delete(0, tk.END)
        for t in types: self.lb_types.insert(tk.END, t)
        
        self.cb_meta_type['values'] = types
        if types:
            if not self.cb_meta_type.get() in types:
                self.cb_meta_type.set(types[0])
            self.load_meta_parts()

    def load_meta_parts(self, event=None):
        v_type = self.cb_meta_type.get()
        self.lb_parts.delete(0, tk.END)
        if not v_type: return
        parts = load_list(f"parts_{v_type}.txt", [f"Деталь_{v_type}"])
        for p in parts: self.lb_parts.insert(tk.END, p)

    def meta_select_item(self, listbox, entry):
        try:
            selection = listbox.curselection()
            if selection:
                val = listbox.get(selection[0])
                entry.delete(0, tk.END)
                entry.insert(0, val)
        except: pass

    def meta_add_type(self):
        val = self.ent_type.get().strip()
        if not val: return
        types = load_list("types.txt", [])
        if val not in types:
            types.append(val)
            save_list("types.txt", types)
            load_list(f"parts_{val.strip()}.txt", [f"Деталь_{val.strip()}"])
            self.load_metadata_tab()
            self.ent_type.delete(0, tk.END)

    def meta_edit_type(self):
        try:
            sel = self.lb_types.curselection()
            if not sel: return
            old_val = self.lb_types.get(sel[0])
            new_val = self.ent_type.get().strip()
            if not new_val: return
            types = load_list("types.txt", [])
            if old_val in types:
                idx = types.index(old_val)
                types[idx] = new_val
                save_list("types.txt", types)
                old_f = f"parts_{old_val}.txt"
                new_f = f"parts_{new_val}.txt"
                if os.path.exists(old_f):
                    os.rename(old_f, new_f)
                self.load_metadata_tab()
        except: pass

    def meta_delete_type(self):
        try:
            sel = self.lb_types.curselection()
            if not sel: return
            val = self.lb_types.get(sel[0])
            types = load_list("types.txt", [])
            if val in types:
                types.remove(val)
                save_list("types.txt", types)
                old_f = f"parts_{val}.txt"
                if os.path.exists(old_f):
                    os.remove(old_f)
                self.load_metadata_tab()
                self.ent_type.delete(0, tk.END)
        except: pass

    def meta_add_part(self):
        v_type = self.cb_meta_type.get()
        val = self.ent_part.get().strip()
        if not v_type or not val: return
        filename = f"parts_{v_type}.txt"
        parts = load_list(filename, [])
        if val not in parts:
            parts.append(val)
            save_list(filename, parts)
            self.load_meta_parts()
            self.ent_part.delete(0, tk.END)

    def meta_edit_part(self):
        try:
            v_type = self.cb_meta_type.get()
            sel = self.lb_parts.curselection()
            if not v_type or not sel: return
            old_val = self.lb_parts.get(sel[0])
            new_val = self.ent_part.get().strip()
            if not new_val: return
            filename = f"parts_{v_type}.txt"
            parts = load_list(filename, [])
            if old_val in parts:
                idx = parts.index(old_val)
                parts[idx] = new_val
                save_list(filename, parts)
                self.load_meta_parts()
        except: pass

    def meta_delete_part(self):
        try:
            v_type = self.cb_meta_type.get()
            sel = self.lb_parts.curselection()
            if not v_type or not sel: return
            val = self.lb_parts.get(sel[0])
            filename = f"parts_{v_type}.txt"
            parts = load_list(filename, [])
            if val in parts:
                parts.remove(val)
                save_list(filename, parts)
                self.load_meta_parts()
                self.ent_part.delete(0, tk.END)
        except: pass

    def on_tab_changed(self, event):
        tab_id = self.notebook.index(self.notebook.select())
        if tab_id == 1:
            self.load_registration_lists()
            self.cb_type.bind("<<ComboboxSelected>>", self.update_part_list)
        elif tab_id == 2:
            self.load_gallery()
        elif tab_id == 3:
            self.load_metadata_tab()

    def play_success_sound(self, text_to_speak):
        def _play():
            if not self.audio_lock.acquire(blocking=False): return
            try:
                bytes_str = text_to_speak.encode('utf-16le')
                b64_str = base64.b64encode(bytes_str).decode('utf-8')

                ps_command = (
                    "$enc = [System.Text.Encoding]::Unicode; "
                    f"$str = $enc.GetString([System.Convert]::FromBase64String('{b64_str}')); "
                    "Add-Type -AssemblyName System.Speech; "
                    "(New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak($str)"
                )
                subprocess.run(["powershell.exe", "-WindowStyle", "Hidden", "-EncodedCommand", 
                                base64.b64encode(ps_command.encode('utf-16le')).decode('utf-8')], 
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except: 
                pass
            finally: 
                self.audio_lock.release()
        threading.Thread(target=_play, daemon=True).start()
   
    def update_frame(self):
        """
        GUI timer НЕ читает камеру.

        CameraManager.reader_thread уже получил кадр и положил его в
        last_frame. Здесь мы только забираем копию и рисуем UI.
        """
        global latest_frame, latest_crop, latest_raw_crop, scan_results

        frame = self.camera_manager.get_frame()

        if frame is None:
            self.set_camera_status(
                "🔎 Камера не подключена — повторный поиск..."
            )

            if (
                not self.camera_scan_in_progress
                and not self.camera_reconnect_scheduled
            ):
                now = time.time()
                if now >= self.camera_retry_after:
                    self.camera_retry_after = now + 3.0
                    self.start_camera_scan()

            self.root.after(100, self.update_frame)
            return

        self.camera_frame_failures = 0

        # Health check without touching VideoCapture.
        is_dark, info = self.camera_manager.note_frame_health(frame)

        if (self.frame_counter + 1) % 120 == 0:
            state = self.camera_manager.get_state()
            camera_log(
                f"[CameraManager] UI frame #{self.frame_counter + 1}; "
                f"index={state['camera_index']}; "
                f"backend={state['backend']}; "
                f"shape={frame.shape}; "
                f"mean={info['mean']:.2f}; std={info['std']:.2f}; "
                f"min={info['min']}; max={info['max']}; "
                f"nonzero={info['nonzero']:.3f}; "
                f"age={self.camera_manager.get_frame_age():.3f}s"
            )

        if is_dark:
            self.camera_dark_frames += 1

            if (
                self.camera_dark_frames >= 60
                and not self.camera_reconnect_scheduled
            ):
                state = self.camera_manager.get_state()

                self.camera_reconnect_scheduled = True
                camera_log(
                    f"[CameraManager] 60 black frames: "
                    f"index={state['camera_index']}, "
                    f"backend={state['backend']}"
                )

                self.set_camera_status(
                    "⚠️ Камера отдает черный поток — восстанавливаем..."
                )

                self.root.after(
                    200,
                    lambda index=state["camera_index"]:
                    self.reconnect_camera_with_next_backend(index),
                )
        else:
            self.camera_dark_frames = 0

        if self.camera_manager.needs_reconnect(
            max_dark_frames=90,
            max_age=3.0,
        ) and not self.camera_reconnect_scheduled:
            state = self.camera_manager.get_state()

            self.camera_reconnect_scheduled = True
            camera_log(
                f"[CameraManager] reconnect required: "
                f"index={state['camera_index']}; "
                f"backend={state['backend']}; "
                f"age={self.camera_manager.get_frame_age():.2f}s"
            )

            self.set_camera_status(
                "⚠️ Видеопоток потерян — переподключение..."
            )

            self.root.after(
                200,
                lambda index=state["camera_index"]:
                self.reconnect_camera_with_next_backend(index),
            )

        if not is_dark:
            state = self.camera_manager.get_state()
            self.active_camera_backend = state["backend"]
            self.set_camera_status(
                f"🟢 Камера #{state['camera_index']} работает "
                f"({state['backend']})"
            )
        else:
            self.set_camera_status(
                "⚠️ Камера отдает очень темный/черный кадр..."
            )

        # Mirror image for the operator.
        frame = cv2.flip(frame, 1)

        h, w = frame.shape[:2]
        zoom_level = 2.0
        size = int(min(h, w) / zoom_level)

        start_y = int(h / 2 - size / 2)
        start_x = int(w / 2 - size / 2)
        start_y = max(0, start_y)
        start_x = max(0, start_x)

        crop = frame[
            start_y:start_y + size,
            start_x:start_x + size
        ]

        if crop is None or crop.size == 0:
            self.root.after(30, self.update_frame)
            return

        with state_lock:
            latest_raw_crop = crop.copy()
            latest_crop = cv2.resize(
                crop,
                (400, 400),
                interpolation=cv2.INTER_AREA,
            )

        self.frame_counter += 1

        if self.frame_counter % 10 == 0 and not self.is_inferencing:
            self.is_inferencing = True

            with state_lock:
                crop_for_inf = latest_crop.copy()

            threading.Thread(
                target=self.recognize_part_thread,
                args=(crop_for_inf,),
                daemon=True,
                name="InferenceWorker",
            ).start()

        try:
            tab_id = self.notebook.index(self.notebook.select())
        except Exception:
            tab_id = 0

        crop_h, crop_w = crop.shape[:2]

        if tab_id == 0:
            if not self.is_detected:
                self.scan_line_y += (
                    int(crop_h * 0.05) * self.scan_line_dir
                )

                if (
                    self.scan_line_y >= crop_h
                    or self.scan_line_y <= 0
                ):
                    self.scan_line_dir *= -1
                    self.scan_line_y = max(
                        0,
                        min(self.scan_line_y, crop_h),
                    )

                cv2.line(
                    crop,
                    (0, self.scan_line_y),
                    (crop_w, self.scan_line_y),
                    (0, 255, 0),
                    3,
                )

                cv2.drawMarker(
                    crop,
                    (crop_w // 2, crop_h // 2),
                    (0, 165, 255),
                    cv2.MARKER_CROSS,
                    40,
                    2,
                )
            else:
                cv2.rectangle(
                    crop,
                    (0, 0),
                    (crop_w - 1, crop_h - 1),
                    (0, 255, 0),
                    8,
                )
        else:
            cv2.drawMarker(
                crop,
                (crop_w // 2, crop_h // 2),
                (255, 255, 255),
                cv2.MARKER_CROSS,
                40,
                2,
            )

        ui_frame = cv2.resize(
            crop,
            (650, 650),
            interpolation=cv2.INTER_LINEAR,
        )

        with state_lock:
            latest_frame = ui_frame.copy()

        try:
            img_tk = ImageTk.PhotoImage(
                image=Image.fromarray(
                    cv2.cvtColor(
                        ui_frame,
                        cv2.COLOR_BGR2RGB,
                    )
                )
            )

            if tab_id == 0:
                self.video_label_scanner.imgtk = img_tk
                self.video_label_scanner.configure(image=img_tk)
                self.sync_scanner_ui()

            elif tab_id == 1:
                self.video_label_register.imgtk = img_tk
                self.video_label_register.configure(image=img_tk)

        except Exception as e:
            camera_log(
                f"Ошибка отображения кадра Tkinter: {repr(e)}"
            )

        self.root.after(30, self.update_frame)

    def recognize_part_thread(self, image_crop):
        try:
            img_pil = Image.fromarray(cv2.cvtColor(image_crop, cv2.COLOR_BGR2RGB))
            with torch.no_grad():
                query_vector = resnet(preprocess(img_pil).unsqueeze(0))[0].detach().numpy().tolist()
            
            with qdrant_lock:
                search_result = client.query_points(collection_name=COLLECTION_NAME, query=query_vector, limit=3)
            
            self.ui_queue.put(lambda: self.process_inference_result(search_result))
        except Exception as e:
            print(f"Ошибка инференса: {e}")
        finally:
            self.is_inferencing = False

    def process_inference_result(self, search_result):
        global scan_results
        with state_lock:
            if hasattr(search_result, 'points') and search_result.points:
                points = search_result.points
                best = points[0]
                
                if best.score >= self.MIN_SCORE:
                    part_name = best.payload.get('name', 'N/A')
                    self.patience_counter = 0 
                    
                    if part_name == self.candidate_name:
                        self.confirm_counter += 1
                    else:
                        self.candidate_name = part_name
                        self.confirm_counter = 1
                        
                    req_frames = self.CONFIRM_FRAMES_FAST if best.score >= 0.85 else self.CONFIRM_FRAMES_SLOW
                    
                    if self.confirm_counter >= req_frames:
                        self.is_detected = True
                        scan_results['part'] = f"Деталь: {part_name}"
                        scan_results['score'] = f"Точность: {int(best.score * 100)}%"
                        scan_results['path_best'] = os.path.join(BASE_DIR, best.payload.get('type'), part_name, f"{best.id}.jpg")
                        
                        if best.score >= 0.85:
                            scan_results['status_text'], scan_results['status_bg'] = "✅ РАСПОЗНАНО", "#10B981"
                        else:
                            scan_results['status_text'], scan_results['status_bg'] = "⚠️ СЛАБОЕ СОВПАДЕНИЕ", "#D97706"
                            
                        if part_name != self.last_spoken_name:
                            self.play_success_sound(part_name)
                            self.last_spoken_name = part_name
                            
                        if len(points) > 1:
                            sim1 = points[1]
                            scan_results['sim1_name'] = sim1.payload.get('name', '---')
                            scan_results['sim1_score'] = f"{int(sim1.score * 100)}%"
                            scan_results['path_sim1'] = os.path.join(BASE_DIR, sim1.payload.get('type'), sim1.payload.get('name'), f"{sim1.id}.jpg")
                        
                        if len(points) > 2:
                            sim2 = points[2]
                            scan_results['sim2_name'] = sim2.payload.get('name', '---')
                            scan_results['sim2_score'] = f"{int(sim2.score * 100)}%"
                            scan_results['path_sim2'] = os.path.join(BASE_DIR, sim2.payload.get('type'), sim2.payload.get('name'), f"{sim2.id}.jpg")
                else:
                    self.handle_detection_loss()
            else:
                self.handle_detection_loss()

    def handle_detection_loss(self):
        global scan_results
        self.candidate_name = None
        self.confirm_counter = 0
        if self.is_detected:
            self.patience_counter += 1
            if self.patience_counter >= self.MAX_PATIENCE:
                self.is_detected = False
                self.last_spoken_name = None
                with state_lock:
                    scan_results = {
                        "part": "Ожидание...", "score": "0%",
                        "status_text": "СКАНИРОВАНИЕ...", "status_bg": "#F59E0B",
                        "sim1_name": "---", "sim1_score": "0%", "path_sim1": "",
                        "sim2_name": "---", "sim2_score": "0%", "path_sim2": "", "path_best": ""
                    }

    def sync_scanner_ui(self):
        with state_lock: 
            sr = scan_results.copy()
        
        if self.lbl_part.cget("text") != sr['part']:
            self.lbl_part.config(text=sr['part'])
            self.lbl_score.config(text=sr['score'])
            self.status_box.config(text=sr['status_text'], bg=sr['status_bg'])
            
            img_best = self.load_ref_image(sr['path_best'], (300, 300))
            self.lbl_best_img.config(image=img_best); self.lbl_best_img.image = img_best
            
            self.lbl_sim1_name.config(text=f"{sr['sim1_name']} ({sr['sim1_score']})")
            img_sim1 = self.load_ref_image(sr['path_sim1'], (200, 200))
            self.lbl_sim1_img.config(image=img_sim1); self.lbl_sim1_img.image = img_sim1
            
            self.lbl_sim2_name.config(text=f"{sr['sim2_name']} ({sr['sim2_score']})")
            img_sim2 = self.load_ref_image(sr['path_sim2'], (200, 200))
            self.lbl_sim2_img.config(image=img_sim2); self.lbl_sim2_img.image = img_sim2

    def on_closing(self):
        try:
            self.camera_manager.shutdown()
        except Exception as e:
            camera_log(f"Ошибка закрытия CameraManager: {repr(e)}")
        self.root.destroy()


# =====================================================================
# FLASK WEB SERVER
# =====================================================================
app_flask = Flask(__name__)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <title>Индустриальный Комплекс Распознавания</title>
    <style>
        body { margin: 0; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #1f1f1f; color: white; display: flex; flex-direction: column; height: 100vh; overflow: hidden; }
        .navbar { display: flex; background-color: #111827; padding: 10px 20px; border-bottom: 2px solid #374151; gap: 15px; }
        .nav-btn { background-color: #374151; color: white; border: none; padding: 10px 20px; font-size: 15px; font-weight: bold; border-radius: 6px; cursor: pointer; transition: 0.2s; }
        .nav-btn.active { background-color: #D32F2F; }
        .nav-btn:hover { background-color: #4B5563; }
        
        .tab-content { display: none; flex: 1; height: calc(100vh - 60px); }
        .tab-content.active { display: flex; }

        .scanner-container { display: flex; width: 100%; height: 100%; }
        .left-panel { width: 350px; background-color: #E5E7EB; color: #333333; padding: 20px; box-sizing: border-box; display: flex; flex-direction: column; }
        .center-panel { flex: 1; display: flex; justify-content: center; align-items: center; background-color: #1f1f1f; }
        .right-panel { width: 300px; background-color: #374151; color: #D1D5DB; padding: 15px 20px; box-sizing: border-box; display: flex; flex-direction: column; }
        .img-best { width: 100%; height: 260px; object-fit: cover; border-radius: 6px; background-color: #4B5563; border: 1px solid #D1D5DB; }
        .img-sim { width: 100%; height: 150px; object-fit: cover; border-radius: 6px; background-color: #4B5563; margin-bottom: 5px; }
        .status-box { margin-top: auto; padding: 12px; font-size: 14px; font-weight: bold; text-align: center; color: white; border-radius: 6px; background-color: #F59E0B; }
        .video-feed { max-width: 100%; max-height: 100%; object-fit: contain; border: 4px solid #374151; border-radius: 8px; }

        .reg-container { display: flex; justify-content: center; align-items: center; width: 100%; height: 100%; background-color: #2b2b2b; }
        .reg-card { background-color: #E5E7EB; color: #333333; padding: 30px; border-radius: 12px; width: 400px; box-shadow: 0 10px 25px rgba(0,0,0,0.5); }
        .form-group { margin-bottom: 20px; }
        label { display: block; font-weight: bold; margin-bottom: 8px; color: #4B5563; }
        select, input { width: 100%; padding: 12px; font-size: 15px; border: 1px solid #D1D5DB; border-radius: 6px; background: white; color: #333; box-sizing: border-box; }
        .btn-save { background-color: #D32F2F; color: white; border: none; padding: 15px; width: 100%; font-size: 16px; font-weight: bold; border-radius: 6px; cursor: pointer; text-transform: uppercase; }
        .btn-save:hover { background-color: #B71C1C; }

        .gallery-container { flex-direction: column; padding: 20px; background-color: #E5E7EB; color: #333; overflow-y: auto; }
        .filters { display: flex; gap: 20px; background: #F9FAFB; padding: 20px; border-radius: 8px; border: 1px solid #D1D5DB; margin-bottom: 20px; }
        .filter-group { flex: 1; display: flex; flex-direction: column; }
        .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 20px; }
        .card { background: white; padding: 15px; border-radius: 8px; border: 1px solid #D1D5DB; text-align: center; }
        .card img { width: 100%; height: 200px; object-fit: cover; border-radius: 6px; margin-bottom: 10px; }
        .btn-delete { background: #D32F2F; color: white; border: none; padding: 10px; width: 100%; border-radius: 6px; font-weight: bold; cursor: pointer; }
        .btn-delete:hover { background: #B71C1C; }

        .meta-container { padding: 30px; background-color: #2b2b2b; color: white; flex-direction: column; overflow-y: auto; }
        .meta-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px; width: 100%; max-width: 900px; margin: 0 auto; }
        .meta-card { background-color: #374151; padding: 20px; border-radius: 8px; display: flex; flex-direction: column; border: 1px solid #4B5563; }
        .meta-card h3 { margin-top: 0; border-bottom: 1px solid #4B5563; padding-bottom: 10px; }
        .meta-list { background: #1f1f1f; border: 1px solid #4B5563; border-radius: 6px; height: 200px; overflow-y: auto; padding: 5px; margin-bottom: 15px; }
        .meta-item { padding: 8px 10px; cursor: pointer; border-radius: 4px; margin-bottom: 2px; }
        .meta-item:hover { background: #374151; }
        .meta-item.selected { background: #D32F2F; font-weight: bold; }
        .meta-actions { display: flex; gap: 10px; margin-top: 10px; }
        .btn-meta { flex: 1; padding: 8px; border: none; border-radius: 4px; font-weight: bold; cursor: pointer; color: white; }
        .btn-meta-add { background: #10B981; }
        .btn-meta-edit { background: #3B82F6; }
        .btn-meta-del { background: #D32F2F; }
    </style>
</head>
<body>
    <div class="navbar">
        <button class="nav-btn active" onclick="switchTab('scanner', this)">📹 Сканер</button>
        <button class="nav-btn" onclick="switchTab('register', this)">📸 Регистрация</button>
        <button class="nav-btn" onclick="switchTab('gallery', this)">🗂️ База Данных</button>
        <button class="nav-btn" onclick="switchTab('metadata', this)">⚙️ Справочники</button>
    </div>

    <div id="scanner" class="tab-content active scanner-container">
        <div class="left-panel">
            <h3>🔍 РЕЗУЛЬТАТ</h3>
            <div style="text-align: center; margin: 10px 0;"><img id="img_best" class="img-best" src="" alt="Нет фото"></div>
            <div id="lbl_part" style="font-size: 16px; font-weight: bold; color: #1D4ED8; margin: 10px 0;">Деталь: ...</div>
            <div id="lbl_score" style="font-size: 14px; font-weight: bold; color: #047857; margin: 10px 0;">Точность: 0%</div>
            <div id="status_box" class="status-box">СКАНИРОВАНИЕ...</div>
        </div>
        <div class="center-panel"><img class="video-feed" src="/video_feed" alt="Видеопоток"></div>
        <div class="right-panel">
            <h3>БЛИЖАЙШИЕ СОВПАДЕНИЯ</h3>
            <div style="text-align: center; margin-bottom: 15px;">
                <img id="img_sim1" class="img-sim" src="" alt="Пусто">
                <div id="lbl_sim1_name" style="font-size: 11px; font-weight: bold;">---</div>
                <div id="lbl_sim1_score" style="font-size: 11px; color: #9CA3AF;">0%</div>
            </div>
            <div style="text-align: center;">
                <img id="img_sim2" class="img-sim" src="" alt="Пусто">
                <div id="lbl_sim2_name" style="font-size: 11px; font-weight: bold;">---</div>
                <div id="lbl_sim2_score" style="font-size: 11px; color: #9CA3AF;">0%</div>
            </div>
        </div>
    </div>

    <div id="register" class="tab-content reg-container">
        <div class="reg-card">
            <h2 style="margin-top:0; color:#111827; border-bottom:2px solid #D1D5DB; padding-bottom:10px;">⚙️ РЕГИСТРАЦИЯ</h2>
            <div class="form-group">
                <label>Тип сборки:</label>
                <select id="reg_type" onchange="loadPartsList()"></select>
            </div>
            <div class="form-group">
                <label>Название детали:</label>
                <input type="text" id="reg_part" list="parts_list" placeholder="Выберите или введите...">
                <datalist id="parts_list"></datalist>
            </div>
            <button class="btn-save" onclick="saveReference()">📸 СОХРАНИТЬ ЭТАЛОН</button>
            <div id="reg_status" style="margin-top:15px; text-align:center; font-weight:bold;"></div>
        </div>
    </div>

    <div id="gallery" class="tab-content gallery-container">
        <h1 style="color:#111827; text-align:center; margin-top:0;">📸 Галерея эталонов</h1>
        <div class="filters">
            <div class="filter-group"><label>Тип</label><select id="gal_type" onchange="onGalleryFilterChange()"></select></div>
            <div class="filter-group"><label>Деталь</label><select id="gal_part" onchange="renderGalleryGrid()" disabled></select></div>
        </div>
        <div class="grid" id="galleryGrid"></div>
    </div>

    <div id="metadata" class="tab-content meta-container">
        <h2 style="text-align: center; margin-top: 0;">⚙️ Управление справочниками</h2>
        <div class="meta-grid" style="grid-template-columns: 1fr; max-width: 900px; margin-bottom: 20px;">
            <div class="meta-card">
                <h3>📹 Выбор камеры</h3>
                <div style="display: flex; gap: 10px; align-items: center; flex-wrap: wrap;">
                    <label style="margin:0;">Доступные камеры:</label>
                    <select id="web_cam_idx" style="width: 110px;"></select>
                    <button class="btn-meta btn-meta-add" onclick="scanWebCameras()">🔎 Найти</button>
                    <button class="btn-meta btn-meta-add" onclick="saveWebCamera()">🔌 Подключить</button>
                    <button class="btn-meta" style="background:#F59E0B;" onclick="openCameraSettings()">🔐 Разрешения Windows</button>
                    <span id="web_camera_status" style="font-weight:bold; color:#D1D5DB;">Проверка...</span>
                </div>
            </div>
        </div>
        <div class="meta-grid">
            <div class="meta-card">
                <h3>Типы (Types)</h3>
                <div id="meta_type_list" class="meta-list"></div>
                <input type="text" id="meta_type_input" placeholder="Новый тип...">
                <div class="meta-actions">
                    <button class="btn-meta btn-meta-add" onclick="metaAdd('type')">Добавить</button>
                    <button class="btn-meta btn-meta-edit" onclick="metaEdit('type')">Изменить</button>
                    <button class="btn-meta btn-meta-del" onclick="metaDelete('type')">Удалить</button>
                </div>
            </div>
            <div class="meta-card">
                <h3>Детали (Parts)</h3>
                <label style="font-size: 13px; margin-bottom: 5px;">Тип сборки:</label>
                <select id="meta_part_type_select" onchange="loadMetaPartsWeb()" style="margin-bottom: 10px;"></select>
                <div id="meta_part_list" class="meta-list"></div>
                <input type="text" id="meta_part_input" placeholder="Новая деталь...">
                <div class="meta-actions">
                    <button class="btn-meta btn-meta-add" onclick="metaAdd('part')">Добавить</button>
                    <button class="btn-meta btn-meta-edit" onclick="metaEdit('part')">Изменить</button>
                    <button class="btn-meta btn-meta-del" onclick="metaDelete('part')">Удалить</button>
                </div>
            </div>
        </div>
    </div>

    <script>
        function switchTab(tabId, btn) {
            document.querySelectorAll('.tab-content').forEach(el => el.style.display = 'none');
            document.querySelectorAll('.nav-btn').forEach(el => el.classList.remove('active'));
            document.getElementById(tabId).style.display = 'flex';
            btn.classList.add('active');
            if(tabId === 'register') loadLists();
            if(tabId === 'gallery') fetchImages();
            if(tabId === 'metadata') loadMetadataWeb();
        }

        setInterval(() => {
            if(document.getElementById('scanner').style.display !== 'none') {
                fetch('/api/scanner_data').then(r => r.json()).then(data => {
                    document.getElementById('lbl_part').innerText = data.part;
                    document.getElementById('lbl_score').innerText = data.score;
                    const sb = document.getElementById('status_box');
                    sb.innerText = data.status_text; sb.style.backgroundColor = data.status_bg;
                    document.getElementById('lbl_sim1_name').innerText = data.sim1_name;
                    document.getElementById('lbl_sim1_score').innerText = data.sim1_score;
                    document.getElementById('lbl_sim2_name').innerText = data.sim2_name;
                    document.getElementById('lbl_sim2_score').innerText = data.sim2_score;
                    document.getElementById('img_best').src = data.path_best ? '/images/' + data.path_best : '';
                    document.getElementById('img_sim1').src = data.path_sim1 ? '/images/' + data.path_sim1 : '';
                    document.getElementById('img_sim2').src = data.path_sim2 ? '/images/' + data.path_sim2 : '';
                });
            }
        }, 200);

        function loadLists() {
            fetch('/api/get_lists').then(r => r.json()).then(data => {
                const t = document.getElementById('reg_type'); t.innerHTML = '';
                data.types.forEach(i => t.innerHTML += `<option value="${i}">${i}</option>`);
                loadPartsList();
            });
        }
        function loadPartsList() {
            const type = document.getElementById('reg_type').value;
            fetch('/api/get_parts?type=' + encodeURIComponent(type)).then(r => r.json()).then(data => {
                const dl = document.getElementById('parts_list'); dl.innerHTML = '';
                data.parts.forEach(i => dl.innerHTML += `<option value="${i}">`);
            });
        }
        function saveReference() {
            const st = document.getElementById('reg_status');
            st.innerText = "⏳ Сохранение..."; st.style.color = "#F59E0B";
            fetch('/api/save', {
                method: 'POST', headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    type: document.getElementById('reg_type').value,
                    part: document.getElementById('reg_part').value
                })
            }).then(r => r.json()).then(data => {
                st.innerText = data.status === 'success' ? "✅ Успешно сохранено!" : "❌ Ошибка: " + data.message;
                st.style.color = data.status === 'success' ? "#10B981" : "#EF4444";
            });
        }

        let allImages = [], catalog = {};
        async function fetchImages() {
            const res = await fetch('/api/images');
            allImages = await res.json();
            catalog = {};
            allImages.forEach(img => {
                if(!catalog[img.type]) catalog[img.type] = new Set();
                catalog[img.type].add(img.part);
            });
            updateGalleryDropdowns();
        }
        function updateGalleryDropdowns() {
            const tSel = document.getElementById('gal_type'), pSel = document.getElementById('gal_part');
            const ct = tSel.value, cp = pSel.value;
            tSel.innerHTML = '<option value="">-- Тип --</option>';
            Object.keys(catalog).sort().forEach(t => tSel.innerHTML += `<option value="${t}">${t}</option>`);
            if(catalog[ct]) tSel.value = ct;
            
            pSel.innerHTML = '<option value="">-- Деталь --</option>'; pSel.disabled = !tSel.value;
            if(tSel.value && catalog[tSel.value]) {
                Array.from(catalog[tSel.value]).sort().forEach(p => pSel.innerHTML += `<option value="${p}">${p}</option>`);
                if(catalog[tSel.value].has(cp)) pSel.value = cp;
            }
            renderGalleryGrid();
        }
        function onGalleryFilterChange() { updateGalleryDropdowns(); }
        function renderGalleryGrid() {
            const t = document.getElementById('gal_type').value, p = document.getElementById('gal_part').value;
            const grid = document.getElementById('galleryGrid'); grid.innerHTML = '';
            if(!t || !p) { grid.innerHTML = '<div style="grid-column:1/-1; text-align:center;">Выберите фильтры сверху</div>'; return; }
            allImages.filter(img => img.type === t && img.part === p).forEach(img => {
                grid.innerHTML += `<div class="card" id="card-${img.id}"><img src="${img.url}?t=${Date.now()}" alt="Эталон"><button class="btn-delete" onclick="deleteImage('${img.id}')">🗑️ Удалить</button></div>`;
            });
        }
        async function deleteImage(id) {
            const card = document.getElementById('card-' + id);
            const btn = card.querySelector('.btn-delete');
            if(!card.dataset.confirmed) {
                card.dataset.confirmed = "true";
                btn.innerText = "⚠️ Подтвердить?";
                btn.style.background = "#B71C1C";
                setTimeout(() => { card.dataset.confirmed = ""; btn.innerText = "🗑️ Удалить"; btn.style.background = "#D32F2F"; }, 3000);
                return;
            }
            const res = await fetch('/api/images/' + id, {method: 'DELETE'});
            if(res.ok) { card.remove(); fetchImages(); }
        }

        let selectedMetaType = null, selectedMetaPart = null;
        async function loadMetadataWeb() {
            const res = await fetch('/api/get_lists');
            const data = await res.json();

            const tList = document.getElementById('meta_type_list'); tList.innerHTML = '';
            const tSelect = document.getElementById('meta_part_type_select'); tSelect.innerHTML = '';
            data.types.forEach(t => {
                tList.innerHTML += `<div class="meta-item ${selectedMetaType === t ? 'selected' : ''}" onclick="selectMetaItem('type', '${t}')">${t}</div>`;
                tSelect.innerHTML += `<option value="${t}" ${selectedMetaType === t ? 'selected' : ''}>${t}</option>`;
            });
            if(!selectedMetaType && data.types.length > 0) selectedMetaType = data.types[0];
            tSelect.value = selectedMetaType;
            loadMetaPartsWeb();
            
            await refreshWebCameras();
        }

        async function refreshWebCameras() {
            try {
                const res = await fetch('/api/cameras');
                const d = await res.json();
                const select = document.getElementById('web_cam_idx');
                const current = String(d.camera_index);
                const values = (d.available && d.available.length) ? d.available.map(String) : [current];
                select.innerHTML = '';
                values.forEach(v => {
                    const o = document.createElement('option');
                    o.value = v; o.textContent = 'Камера #' + v;
                    select.appendChild(o);
                });
                if(values.includes(current)) select.value = current;
                document.getElementById('web_camera_status').textContent = d.active ? ('🟢 Подключена #' + current) : ('⚠️ ' + (d.error || 'Не подключена'));
            } catch(e) {
                document.getElementById('web_camera_status').textContent = '❌ Нет связи с приложением';
            }
        }

        async function scanWebCameras() {
            document.getElementById('web_camera_status').textContent = '🔎 Идет поиск...';
            await fetch('/api/scan_cameras', {method:'POST'});
            let attempts = 0;
            const timer = setInterval(async () => {
                await refreshWebCameras();
                attempts++;
                if(attempts >= 30) clearInterval(timer);
            }, 1000);
        }

        async function openCameraSettings() {
            await fetch('/api/open_camera_settings', {method:'POST'});
            alert('Открыты настройки Windows. Включите доступ к камере для классических приложений.');
        }

        async function saveWebCamera() {
            const idx = document.getElementById('web_cam_idx').value;
            const res = await fetch('/api/set_camera', {
                method: 'POST', headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({camera_index: parseInt(idx)})
            });
            const data = await res.json();
            if(data.status === 'success') {
                alert('Камера #' + idx + ' подключена!');
            } else {
                alert('Не удалось подключить камеру #' + idx + '. Смотрите camera_debug.log');
            }
            await refreshWebCameras();
        }

        async function loadMetaPartsWeb() {
            const type = document.getElementById('meta_part_type_select').value;
            selectedMetaType = type;
            const res = await fetch('/api/get_parts?type=' + encodeURIComponent(type));
            const data = await res.json();
            const pList = document.getElementById('meta_part_list'); pList.innerHTML = '';
            data.parts.forEach(p => {
                pList.innerHTML += `<div class="meta-item ${selectedMetaPart === p ? 'selected' : ''}" onclick="selectMetaItem('part', '${p}')">${p}</div>`;
            });
        }

        function selectMetaItem(category, val) {
            if(category === 'type') { selectedMetaType = val; document.getElementById('meta_type_input').value = val; loadMetadataWeb(); }
            if(category === 'part') { selectedMetaPart = val; document.getElementById('meta_part_input').value = val; loadMetaPartsWeb(); }
        }

        async function metaAdd(category) {
            let val = '';
            if(category === 'type') val = document.getElementById('meta_type_input').value.trim();
            if(category === 'part') val = document.getElementById('meta_part_input').value.trim();
            if(!val) return;

            await fetch('/api/meta/add', {
                method: 'POST', headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({category: category, value: val, type: selectedMetaType})
            });
            loadMetadataWeb();
        }

        async function metaEdit(category) {
            let oldVal = (category === 'type' ? selectedMetaType : selectedMetaPart);
            let newVal = '';
            if(category === 'type') newVal = document.getElementById('meta_type_input').value.trim();
            if(category === 'part') newVal = document.getElementById('meta_part_input').value.trim();
            if(!oldVal || !newVal) return;

            await fetch('/api/meta/edit', {
                method: 'POST', headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({category: category, old_value: oldVal, new_value: newVal, type: selectedMetaType})
            });
            if(category === 'type') selectedMetaType = newVal;
            if(category === 'part') selectedMetaPart = newVal;
            loadMetadataWeb();
        }

        async function metaDelete(category) {
            let val = (category === 'type' ? selectedMetaType : selectedMetaPart);
            if(!val) return;

            await fetch('/api/meta/delete', {
                method: 'POST', headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({category: category, value: val, type: selectedMetaType})
            });
            if(category === 'type') selectedMetaType = null;
            if(category === 'part') selectedMetaPart = null;
            loadMetadataWeb();
        }
    </script>
</body>
</html>
"""

@app_flask.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)

def gen_mjpeg():
    global latest_frame
    while True:
        with state_lock:
            frame_to_send = latest_frame.copy() if latest_frame is not None else None
        if frame_to_send is not None:
            ret, buffer = cv2.imencode('.jpg', frame_to_send)
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        time.sleep(0.03)

@app_flask.route("/video_feed")
def video_feed():
    return Response(gen_mjpeg(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app_flask.route("/api/scanner_data")
def get_scanner_data():
    with state_lock:
        return jsonify(scan_results)

@app_flask.route("/api/get_lists")
def get_lists_api():
    types = load_list("types.txt", ["metiz", "bigdetail"])
    return jsonify({"types": types})

@app_flask.route("/api/get_parts")
def get_parts_api():
    v_type = request.args.get('type', '')
    parts = load_list(f"parts_{v_type}.txt", [f"Деталь_{v_type}"])
    return jsonify({"parts": parts})

@app_flask.route("/api/get_camera")
def get_camera_api():
    application = globals().get("app")
    if application:
        state = application.camera_manager.get_state()
        return jsonify({
            "camera_index": state["camera_index"],
            "available": state["available"],
            "active": state["active"],
            "backend": state["backend"],
            "error": state["status"] or state["error"],
        })

    return jsonify({
        "camera_index": load_camera_index(),
        "available": [],
        "active": False,
        "backend": "",
        "error": "Приложение еще запускается",
    })

@app_flask.route("/api/cameras")
def cameras_api():
    """Состояние CameraManager без доступа Flask к VideoCapture."""
    if "app" not in globals():
        return jsonify({
            "available": [],
            "camera_index": load_camera_index(),
            "active": False,
            "backend": "",
            "error": "Приложение еще запускается",
        })

    state = app.camera_manager.get_state()
    return jsonify({
        "available": state["available"],
        "camera_index": state["camera_index"],
        "active": state["active"],
        "backend": state["backend"],
        "error": state["status"] or state["error"],
    })

@app_flask.route("/api/scan_cameras", methods=["POST"])
def scan_cameras_api():
    try:
        app.start_camera_scan()
        return jsonify({"status": "scanning"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app_flask.route("/api/set_camera", methods=["POST"])
def set_camera_api():
    data = request.json or {}
    idx = int(data.get("camera_index", 0))
    ok = app.switch_camera(idx, show_message=False) if "app" in globals() else False
    return jsonify({"status": "success" if ok else "error", "camera_index": idx})

@app_flask.route("/api/open_camera_settings", methods=["POST"])
def open_camera_settings_api():
    try:
        ok = app.open_windows_camera_settings() if "app" in globals() else False
        return jsonify({"status": "success" if ok else "error"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app_flask.route("/api/meta/add", methods=["POST"])
def meta_add_api():
    data = request.json
    cat, val, v_type = data.get('category'), data.get('value'), data.get('type')
    if cat == 'type':
        types = load_list("types.txt", [])
        if val not in types:
            types.append(val)
            save_list("types.txt", types)
            load_list(f"parts_{val}.txt", [f"Деталь_{val}"])
    elif cat == 'part':
        if v_type:
            f_name = f"parts_{v_type}.txt"
            parts = load_list(f_name, [])
            if val not in parts:
                parts.append(val)
                save_list(f_name, parts)
    return jsonify({"status": "success"})

@app_flask.route("/api/meta/edit", methods=["POST"])
def meta_edit_api():
    data = request.json
    cat, old_v, new_v, v_type = data.get('category'), data.get('old_value'), data.get('new_value'), data.get('type')
    if cat == 'type':
        types = load_list("types.txt", [])
        if old_v in types:
            types[types.index(old_v)] = new_v
            save_list("types.txt", types)
            old_f = f"parts_{old_v}.txt"
            new_f = f"parts_{new_v}.txt"
            if os.path.exists(old_f):
                os.rename(old_f, new_f)
    elif cat == 'part':
        if v_type:
            f_name = f"parts_{v_type}.txt"
            parts = load_list(f_name, [])
            if old_v in parts:
                parts[parts.index(old_v)] = new_v
                save_list(f_name, parts)
    return jsonify({"status": "success"})

@app_flask.route("/api/meta/delete", methods=["POST"])
def meta_delete_api():
    data = request.json
    cat, val, v_type = data.get('category'), data.get('value'), data.get('type')
    if cat == 'type':
        types = load_list("types.txt", [])
        if val in types:
            types.remove(val)
            save_list("types.txt", types)
            f_name = f"parts_{val}.txt"
            if os.path.exists(f_name):
                os.remove(f_name)
    elif cat == 'part':
        if v_type:
            f_name = f"parts_{v_type}.txt"
            parts = load_list(f_name, [])
            if val in parts:
                parts.remove(val)
                save_list(f_name, parts)
    return jsonify({"status": "success"})

@app_flask.route("/api/save", methods=["POST"])
def save_api():
    global latest_crop, latest_raw_crop
    data = request.json
    v_type, part = data.get('type'), data.get('part')
    
    with state_lock:
        local_crop = latest_crop.copy() if latest_crop is not None else None
        local_raw = latest_raw_crop.copy() if latest_raw_crop is not None else None

    if local_crop is None or local_raw is None:
        return jsonify({"status": "error", "message": "Нет кадра с камеры"})
        
    types = load_list("types.txt", [])
    if v_type not in types: save_list("types.txt", types + [v_type])
    parts = load_list(f"parts_{v_type}.txt", [])
    if part not in parts: save_list(f"parts_{v_type}.txt", parts + [part])

    try:
        img_pil = Image.fromarray(cv2.cvtColor(local_crop, cv2.COLOR_BGR2RGB))
        with torch.no_grad():
            vector = resnet(preprocess(img_pil).unsqueeze(0))[0].detach().numpy().tolist()

        point_id = str(uuid.uuid4())
        save_dir = os.path.join(BASE_DIR, v_type, part)
        os.makedirs(save_dir, exist_ok=True)
        cv2.imwrite(os.path.join(save_dir, f"{point_id}.jpg"), local_raw)

        with qdrant_lock:
            client.upsert(
                COLLECTION_NAME, 
                [PointStruct(id=point_id, vector=vector, payload={"name": part, "type": v_type, "group_id": f"{v_type}_{part}"})]
            )
            
        return jsonify({"status": "success"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app_flask.route("/api/images")
def list_images_api():
    images = []
    if not os.path.exists(BASE_DIR): return jsonify(images)
    for v_type in os.listdir(BASE_DIR):
        t_path = os.path.join(BASE_DIR, v_type)
        if not os.path.isdir(t_path): continue
        for part in os.listdir(t_path):
            p_path = os.path.join(t_path, part)
            if not os.path.isdir(p_path): continue
            for f in os.listdir(p_path):
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    rel_path = f"{v_type}/{part}/{f}"
                    img_id = base64.urlsafe_b64encode(rel_path.encode('utf-8')).decode('utf-8')
                    images.append({"id": img_id, "url": f"/images/{rel_path}", "type": v_type, "part": part})
    return jsonify(images)

@app_flask.route("/api/images/<img_id>", methods=["DELETE"])
def delete_image_api(img_id):
    try:
        rel_path = base64.urlsafe_b64decode(img_id.encode('utf-8')).decode('utf-8')
        full_path = os.path.join(BASE_DIR, rel_path)
        if os.path.exists(full_path):
            point_id = os.path.splitext(os.path.basename(full_path))[0]
            try: 
                with qdrant_lock:
                    client.delete(collection_name=COLLECTION_NAME, points_selector=PointIdsList(points=[point_id]))
            except: pass
            os.remove(full_path)
            return jsonify({"status": "success"})
        return jsonify({"status": "error", "message": "Не найден"}), 404
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app_flask.route("/images/<path:filename>")
def serve_image(filename):
    return send_from_directory(BASE_DIR, filename)

def run_flask():
    app_flask.run(host="0.0.0.0", port=HOST_PORT, debug=False, use_reloader=False)

if __name__ == "__main__":
    print(f"\n=== ИНДУСТРИАЛЬНЫЙ КОМПЛЕКС ЗАПУЩЕН ===")
    print(f"🔗 Web-интерфейс доступен по адресу: http://localhost:{HOST_PORT}\n")

    root = tk.Tk()
    app = IndustrialVisionApp(root)
    globals()["app"] = app

    # Flask starts after the application object exists.
    threading.Thread(
        target=run_flask,
        daemon=True,
        name="FlaskServer",
    ).start()

    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()

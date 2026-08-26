import io
import sys
import os
import codecs
import cv2
import torch
import uuid
import base64
import time
import threading
import subprocess
import queue
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk, ImageDraw
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct, PointIdsList
from torchvision.models import resnet50, ResNet50_Weights

# --- ИМПОРТЫ ДЛЯ FLASK ---
from flask import Flask, Response, render_template_string, jsonify, request, send_from_directory

# Безопасная установка кодировки для stdout/stderr
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

# --- SYSTEM SETTINGS ---
#
# PyInstaller removes the temporary directory used by a --onefile build after
# exit. User-created data must therefore live in a permanent writable folder.
APP_NAME = "IndustrialVision"
COLLECTION_NAME = "parts_resnet50"
MODEL_FILENAME = "resnet50-11ad3fa6.pth"

if getattr(sys, 'frozen', False):
    # Resources bundled by PyInstaller are unpacked here for the current run.
    RESOURCE_DIR = sys._MEIPASS
    INSTALL_DIR = os.path.dirname(sys.executable)
else:
    RESOURCE_DIR = os.path.dirname(os.path.abspath(__file__))
    INSTALL_DIR = RESOURCE_DIR

def resource_path(*parts):
    """Return the path to a read-only application resource."""
    return os.path.join(RESOURCE_DIR, *parts)

def application_data_dir():
    """Return a stable, user-writable folder for application data."""
    if sys.platform == "win32":
        return os.path.join(
            os.environ.get("LOCALAPPDATA", os.path.expanduser("~")), APP_NAME
        )
    return os.path.join(INSTALL_DIR, ".industrial_vision_data")

DATA_DIR = application_data_dir()
BASE_PATH = DATA_DIR  # Compatibility name for the rest of the module.
BASE_DIR = os.path.join(DATA_DIR, "reference_images")
QDRANT_DIR = os.path.join(DATA_DIR, "qdrant_storage")
CAMERA_INDEX_FILE = os.path.join(DATA_DIR, "camera_index.txt")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(BASE_DIR, exist_ok=True)
os.makedirs(QDRANT_DIR, exist_ok=True)
# All legacy relative paths (types.txt, parts_*.txt) now resolve here.
os.chdir(DATA_DIR)
HOST_PORT = 5000

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

# --- LOCAL QDRANT AND RESNET50 INITIALIZATION ---
def load_resnet50_feature_extractor():
    """Load packaged ResNet weights without requiring internet access."""
    model = resnet50(weights=None)
    bundled_weights = resource_path("models", MODEL_FILENAME)

    if os.path.isfile(bundled_weights):
        try:
            state_dict = torch.load(
                bundled_weights, map_location="cpu", weights_only=True
            )
        except TypeError:
            # Compatibility with older PyTorch releases.
            state_dict = torch.load(bundled_weights, map_location="cpu")
        model.load_state_dict(state_dict)
    elif getattr(sys, "frozen", False):
        raise RuntimeError(
            "The packaged ResNet50 model is missing. "
            "Build the application with build_windows.bat."
        )
    else:
        # Source-code development may use Torch's normal cache/download path.
        model = resnet50(weights=ResNet50_Weights.DEFAULT)

    model.fc = torch.nn.Identity()
    model.eval()
    return model

print("⚙️ Инициализация локального Qdrant и ResNet50...")
client = QdrantClient(path=QDRANT_DIR)
if not client.collection_exists(COLLECTION_NAME):
    client.create_collection(
        COLLECTION_NAME, 
        vectors_config=VectorParams(size=2048, distance=Distance.COSINE)
    )
resnet = load_resnet50_feature_extractor()
preprocess = ResNet50_Weights.DEFAULT.transforms()
print("✅ ИИ и База данных успешно инициализированы!")

def load_list(filename, default):
    if not os.path.exists(filename):
        with open(filename, "w", encoding="utf-8") as f:
            f.write("\n".join(default))
        return default
    with open(filename, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

def save_list(filename, data_list):
    with open(filename, "w", encoding="utf-8") as f:
        f.write("\n".join(data_list))

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

        self.init_camera()

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

        if self.cap and self.cap.isOpened():
            self.update_frame()

    def init_camera(self):
        global CAMERA_INDEX
        CAMERA_INDEX = load_camera_index()
        self.cap = cv2.VideoCapture(CAMERA_INDEX)
        if not self.cap.isOpened():
            for idx in [0, 1, 2]:
                if idx != CAMERA_INDEX:
                    self.cap = cv2.VideoCapture(idx)
                    if self.cap.isOpened():
                        break
        
        if self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            try:
                self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            except:
                pass
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
        else:
            print("❌ Ошибка: Камера не найдена!")

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
        tk.Label(cam_frame, text="📹 Выбор индекса камеры:", font=("Arial", 11, "bold"), bg="#374151", fg="white").pack(side=tk.LEFT, padx=10)
        self.cb_camera_idx = ttk.Combobox(cam_frame, values=["0", "1", "2", "3"], font=("Arial", 11), width=5, state="readonly")
        self.cb_camera_idx.pack(side=tk.LEFT, padx=10)
        self.cb_camera_idx.set(str(load_camera_index()))
        tk.Button(cam_frame, text="💾 Сохранить и перезапустить камеру", bg="#10B981", fg="white", font=("Arial", 10, "bold"), command=self.apply_camera_index).pack(side=tk.LEFT, padx=15)

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
            save_camera_index(new_idx)
            if self.cap and self.cap.isOpened():
                self.cap.release()
            self.init_camera()
            messagebox.showinfo("Успех", f"Камера переключена на индекс: {new_idx}")
            print(f"✅ Камера переключена на индекс: {new_idx}")
        except Exception as e:
            print(f"Ошибка переключения камеры: {e}")

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
        global latest_frame, latest_crop, latest_raw_crop, scan_results
        
        if not self.cap or not self.cap.isOpened():
            self.root.after(100, self.update_frame)
            return

        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            zoom_level = 2.0
            size = int(min(h, w) / zoom_level)
            start_y, start_x = int(h/2 - size/2), int(w/2 - size/2)
            start_y, start_x = max(0, start_y), max(0, start_x)

            crop = frame[start_y:start_y+size, start_x:start_x+size]
            
            with state_lock:
                latest_raw_crop = crop.copy()
                latest_crop = cv2.resize(crop, (400, 400), interpolation=cv2.INTER_AREA)
            
            self.frame_counter += 1
            if self.frame_counter % 10 == 0 and not self.is_inferencing:
                self.is_inferencing = True
                with state_lock:
                    crop_for_inf = latest_crop.copy()
                threading.Thread(target=self.recognize_part_thread, args=(crop_for_inf,), daemon=True).start()

            tab_id = self.notebook.index(self.notebook.select())
            crop_h, crop_w = crop.shape[:2]
            
            if tab_id == 0:
                if not self.is_detected:
                    self.scan_line_y += int(crop_h * 0.05) * self.scan_line_dir
                    if self.scan_line_y >= crop_h or self.scan_line_y <= 0:
                        self.scan_line_dir *= -1
                        self.scan_line_y = max(0, min(self.scan_line_y, crop_h))
                    cv2.line(crop, (0, self.scan_line_y), (crop_w, self.scan_line_y), (0, 255, 0), 3)
                    cv2.drawMarker(crop, (crop_w//2, crop_h//2), (0, 165, 255), cv2.MARKER_CROSS, 40, 2)
                else:
                    cv2.rectangle(crop, (0, 0), (crop_w-1, crop_h-1), (0, 255, 0), 8)
            else:
                cv2.drawMarker(crop, (crop_w//2, crop_h//2), (255, 255, 255), cv2.MARKER_CROSS, 40, 2)

            ui_frame = cv2.resize(crop, (650, 650), interpolation=cv2.INTER_LINEAR)
            with state_lock: 
                latest_frame = ui_frame.copy()
            
            img_tk = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(ui_frame, cv2.COLOR_BGR2RGB)))
            
            if tab_id == 0:
                self.video_label_scanner.imgtk = img_tk
                self.video_label_scanner.configure(image=img_tk)
                self.sync_scanner_ui()
            elif tab_id == 1:
                self.video_label_register.imgtk = img_tk
                self.video_label_register.configure(image=img_tk)
                
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
        if self.cap and self.cap.isOpened(): 
            self.cap.release()
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
                <div style="display: flex; gap: 10px; align-items: center;">
                    <label style="margin:0;">Индекс камеры:</label>
                    <select id="web_cam_idx" style="width: 100px;"><option value="0">0</option><option value="1">1</option><option value="2">2</option><option value="3">3</option></select>
                    <button class="btn-meta btn-meta-add" onclick="saveWebCamera()">Установить</button>
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
            
            fetch('/api/get_camera').then(r => r.json()).then(d => {
                document.getElementById('web_cam_idx').value = d.camera_index;
            });
        }

        async function saveWebCamera() {
            const idx = document.getElementById('web_cam_idx').value;
            await fetch('/api/set_camera', {
                method: 'POST', headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({camera_index: parseInt(idx)})
            });
            alert("Камера переключена!");
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
    return jsonify({"camera_index": load_camera_index()})

@app_flask.route("/api/set_camera", methods=["POST"])
def set_camera_api():
    data = request.json
    idx = data.get("camera_index", 0)
    save_camera_index(idx)
    return jsonify({"status": "success"})

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
    threading.Thread(target=run_flask, daemon=True).start()
    
    print(f"\n=== ИНДУСТРИАЛЬНЫЙ КОМПЛЕКС ЗАПУЩЕН ===")
    print(f"🔗 Web-интерфейс доступен по адресу: http://localhost:{HOST_PORT}\n")

    root = tk.Tk()
    app = IndustrialVisionApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()

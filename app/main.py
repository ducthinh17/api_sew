#!/usr/bin/env python3
import os
import io
import uuid # Để tạo tên file tạm thời duy nhất
from typing import Dict, Optional, Any
from contextlib import asynccontextmanager # Cho lifespan
# --- Các import từ script gốc của bạn ---
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np

import torchvision
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torchvision.models import resnet18, ResNet18_Weights
import torchvision.transforms as transforms
from sklearn.preprocessing import LabelEncoder

# --- Pillow HEIF import ---
_HEIC_SUPPORT_PILLOW_INTERNAL = False
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
    _HEIC_SUPPORT_PILLOW_INTERNAL = True
    print("INFO: pillow_heif imported and HEIC opener registered successfully.")
except ImportError:
    print("WARN: pillow_heif library not found. HEIC file support will be unavailable.")
except Exception as e_heif_reg:
    _HEIC_SUPPORT_PILLOW_INTERNAL = True # Giả sử import thành công nhưng có vấn đề với register_opener
    print(f"INFO: pillow_heif imported. Note on opener registration: {e_heif_reg}")

# --- FastAPI và các thành phần liên quan ---
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# ==== A. ĐỊNH NGHĨA CÁC THÀNH PHẦN CẦN THIẾT ====

# 1. Định nghĩa Class MultiTaskModel (Không thay đổi)
class MultiTaskModel(nn.Module):
    """Model Multi-Task với backbone và các head riêng cho từng task."""
    def __init__(self, backbone_name="efficientnet_b0", pretrained=True, num_classes_per_task_dict=None):
        super().__init__()
        self.backbone_name = backbone_name
        if num_classes_per_task_dict is None or not num_classes_per_task_dict:
            raise ValueError("num_classes_per_task_dict không được rỗng và phải được cung cấp.")
        self.task_names = list(num_classes_per_task_dict.keys())

        num_backbone_features = 0
        weights_eff = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained and backbone_name == "efficientnet_b0" else None
        weights_res = ResNet18_Weights.IMAGENET1K_V1 if pretrained and backbone_name == "resnet18" else None

        if backbone_name == "efficientnet_b0":
            self.backbone = efficientnet_b0(weights=weights_eff)
            num_backbone_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
        elif backbone_name == "resnet18":
            self.backbone = resnet18(weights=weights_res)
            num_backbone_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        else:
            raise ValueError(f"Backbone '{backbone_name}' không được hỗ trợ.")

        if num_backbone_features <= 0:
            raise ValueError(f"Không thể xác định số features đầu ra cho backbone {backbone_name}.")

        self.heads = nn.ModuleDict()
        for task_name, num_classes in num_classes_per_task_dict.items():
            if num_classes <= 0:
                continue
            if task_name == "level":
                self.heads[task_name] = nn.Sequential(
                    nn.Linear(num_backbone_features, num_backbone_features // 2),
                    nn.BatchNorm1d(num_backbone_features // 2),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(num_backbone_features // 2, num_classes)
                )
            elif task_name == "sub_category":
                self.heads[task_name] = nn.Sequential(
                    nn.Linear(num_backbone_features, num_backbone_features // 4),
                    nn.ReLU(),
                    nn.Dropout(0.25),
                    nn.Linear(num_backbone_features // 4, num_classes)
                )
            else: # For 'category' and 'orientation'
                self.heads[task_name] = nn.Linear(num_backbone_features, num_classes)
        if not self.heads:
            raise ValueError("Không có head nào được khởi tạo cho model.")

    def forward(self, x):
        features = self.backbone(x)
        if features.ndim == 4: # Ensure features are flattened
            features = features.view(features.size(0), -1)
        outputs = {task_name: head(features) for task_name, head in self.heads.items()}
        return outputs

# 2. Định nghĩa hàm full_process_image (Không thay đổi)
def full_process_image(original_image_path: str, target_size_tuple=(224, 224), heic_is_supported_by_server: bool = False):
    try:
        img = None
        file_ext = os.path.splitext(original_image_path)[1].lower()

        if file_ext in (".heic", ".heif"):
            if not heic_is_supported_by_server:
                print(f"ERROR [full_process_image]: Server HEIC support is disabled or unavailable for {original_image_path}.")
                return None
            try:
                img = Image.open(original_image_path) # Pillow uses registered opener
            except Exception as e_heic:
                print(f"ERROR [full_process_image]: Failed to read HEIC {original_image_path} with Pillow/pillow_heif: {e_heic}.")
                return None
        else:
            img = Image.open(original_image_path)

        img = img.convert("RGB")
        img = img.resize(target_size_tuple, Image.Resampling.BILINEAR)
        return np.array(img)
    except FileNotFoundError:
        print(f"ERROR [full_process_image]: File not found: {original_image_path}")
        return None
    except Exception as e:
        print(f"ERROR [full_process_image]: Could not process {original_image_path}. Error: {e}")
        return None

# 3. Định nghĩa `config` và `UPLOAD_DIR` (Không thay đổi)
MODEL_PATH_SERVER = os.getenv("MODEL_PATH", "./best_multitask_model_generic_last.pth")
UPLOAD_DIR        = "/tmp/image_uploads" # Thư mục tạm trên Render

config = {
    'img_size': 224,
    'target_columns': ['category', 'sub_category', 'level', 'orientation'],
    'num_classes_per_task': {
        'category': 2,
        'sub_category': 7,
        'level': 5,
        'orientation': 2
    },
    'mtl_model_name': 'efficientnet_b0',
    'mtl_pretrained_backbone': True,
}

# 4. Định nghĩa `label_encoder_map_mtl` (Không đổi)
label_encoder_map_mtl = {}
le_category = LabelEncoder(); le_category.fit(['cruve', 'straight']); label_encoder_map_mtl['category'] = le_category
le_sub_category = LabelEncoder(); le_sub_category.fit(['circle', 'cruve', 'line', 'spiral', 'square', 'wave', 'zigzag']); label_encoder_map_mtl['sub_category'] = le_sub_category
le_level = LabelEncoder(); le_level.fit(['Level 1', 'Level 2', 'Level 3', 'Level 4', 'Level 5']); label_encoder_map_mtl['level'] = le_level
le_orientation = LabelEncoder(); le_orientation.fit(['back', 'front']); label_encoder_map_mtl['orientation'] = le_orientation

# --- B. CÁC HÀM TIỆN ÍCH --- (Không thay đổi)

def load_trained_model_server(model_path: str, model_class_def: Any, config_for_model: Dict, device_to_load: str):
    print(f"INFO: Attempting to load model from: {model_path} (PWD: {os.getcwd()})")

    model = model_class_def(
        backbone_name=config_for_model["mtl_model_name"],
        pretrained=config_for_model["mtl_pretrained_backbone"],
        num_classes_per_task_dict=config_for_model["num_classes_per_task"]
    )
    try:
        if not os.path.exists(model_path):
            print(f"ERROR: Model file not found at {model_path}.")
            return None

        print(f"INFO: Loading model state_dict from validated path: {model_path}")
        state_dict = torch.load(model_path, map_location=torch.device(device_to_load))
        model.load_state_dict(state_dict, strict=True)
        print("INFO: Successfully loaded model weights (strict=True).")
    except RuntimeError as e_strict_true:
        print(f"ERROR: Failed to load model weights with strict=True (mismatched keys, etc.). Error: {e_strict_true}")
        return None
    except Exception as e_load:
        print(f"ERROR: An unexpected error occurred while loading the model: {e_load}")
        return None

    model.to(device_to_load)
    model.eval()
    return model

def preprocess_single_image_for_mtl_server(original_image_path: str, target_size: int, ptorch_transforms: Any, heic_is_supported: bool):
    if ptorch_transforms is None:
        print("ERROR [preprocess_single_image_for_mtl_server]: PyTorch transforms missing.")
        return None

    processed_np_array = full_process_image(
        original_image_path,
        target_size_tuple=(target_size, target_size),
        heic_is_supported_by_server=heic_is_supported
    )
    if processed_np_array is None: return None

    try:
        pil_image = Image.fromarray(processed_np_array)
        img_tensor = ptorch_transforms(pil_image)
        return img_tensor
    except Exception as e_tf:
        print(f"ERROR [preprocess_single_image_for_mtl_server]: Image to Tensor or PyTorch transform failed: {e_tf}")
        return None

# ==== GLOBAL APP STATE ==== (Không thay đổi)
app_state: Dict[str, Any] = {
    "inference_model": None,
    "inference_transforms": None,
    "device": 'cuda' if torch.cuda.is_available() else 'cpu',
    "upload_dir_ready": False,
    "heic_support_active": _HEIC_SUPPORT_PILLOW_INTERNAL
}

# ==== LIFESPAN EVENTS ==== (Không thay đổi)
@asynccontextmanager
async def lifespan(app_lifespan_param: FastAPI):
    global app_state

    app_state["device"] = config.get('device', app_state["device"])
    print(f"INFO: Server starting up. Using device: {app_state['device']}")
    print(f"INFO: HEIC Support via Pillow-HEIF: {app_state['heic_support_active']}")

    print(f"INFO: Ensuring UPLOAD_DIR ({UPLOAD_DIR}) is ready.")
    try:
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        if os.access(UPLOAD_DIR, os.W_OK):
            app_state["upload_dir_ready"] = True
            print(f"INFO: UPLOAD_DIR is ready at {UPLOAD_DIR}.")
        else:
            print(f"CRITICAL: UPLOAD_DIR ({UPLOAD_DIR}) was created but is not writable. File uploads will fail.")
    except Exception as e:
        print(f"CRITICAL: Could not create/access UPLOAD_DIR ({UPLOAD_DIR}). Error: {e}. File uploads will fail.")

    try:
        app_state["inference_transforms"] = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        print("INFO: Inference transforms created successfully.")
    except Exception as e_init_transforms:
        print(f"CRITICAL ERROR: Could not create inference_transforms: {e_init_transforms}")

    if app_state["inference_transforms"]:
        print(f"Attempting to load model. Expected path: {MODEL_PATH_SERVER}")
        loaded_model = load_trained_model_server(
            MODEL_PATH_SERVER,
            MultiTaskModel,
            config,
            app_state['device']
        )
        if loaded_model:
            app_state["inference_model"] = loaded_model
            print("INFO: Model loaded successfully. Server is ready.")
        else:
            print("CRITICAL ERROR: Failed to load the model. Predict endpoint will not function.")
    else:
        print("WARN: Model loading skipped because inference_transforms failed to initialize.")

    print("INFO: Application startup actions complete.")
    yield
    print("INFO: Server shutting down.")

# ==== FASTAPI APP ==== (Không thay đổi)
app = FastAPI(
    title="Multi-Task Image Classification Server (Render)",
    description="Upload an image for multi-task classification.",
    version="1.2", # Version bump
    lifespan=lifespan
)

# ==== CORS MIDDLEWARE ==== (Không thay đổi)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==== Pydantic Model cho response body (Không đổi) ====
class PredictionDetail(BaseModel):
    label: str
    confidence: str

class PredictionResponse(BaseModel):
    filename: str
    predictions: Dict[str, PredictionDetail]
    error: Optional[str] = None

# ==== Health Check ==== (Không thay đổi)
@app.get("/health")
async def health():
    return {
        "status": "ok" if app_state["inference_model"] and app_state["inference_transforms"] and app_state["upload_dir_ready"] else "degraded",
        "model_status": "loaded" if app_state["inference_model"] else "not_loaded_or_failed",
        "transforms_status": "initialized" if app_state["inference_transforms"] else "not_initialized",
        "upload_dir_status": "ready" if app_state["upload_dir_ready"] else "not_ready",
        "device": app_state["device"],
        "heic_support": app_state["heic_support_active"]
    }

# ==== Prediction Endpoint (ĐÃ CẬP NHẬT) ====
@app.post("/predict", response_model=PredictionResponse)
async def predict_image(file: UploadFile = File(...)):
    if not app_state.get("inference_model"):
        raise HTTPException(status_code=503, detail="Model is not loaded. Cannot process predictions.")
    if not app_state.get("inference_transforms"):
        raise HTTPException(status_code=503, detail="Image transforms are not initialized. Cannot process predictions.")
    if not app_state.get("upload_dir_ready"):
        raise HTTPException(status_code=503, detail="Server temporary storage is not ready. Cannot process uploads.")

    original_filename = file.filename
    file_extension = os.path.splitext(original_filename)[1].lower()
    temp_filename = f"{uuid.uuid4()}{file_extension}"
    temp_image_path = os.path.join(UPLOAD_DIR, temp_filename)

    try:
        with open(temp_image_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        if file_extension in (".heic", ".heif") and not app_state["heic_support_active"]:
            msg = f"Cannot process HEIC/HEIF: '{original_filename}'. Server lacks HEIC support."
            print(f"ERROR: {msg}")
            raise HTTPException(status_code=400, detail=msg)

        img_tensor = preprocess_single_image_for_mtl_server(
            temp_image_path,
            config['img_size'],
            app_state["inference_transforms"],
            heic_is_supported=app_state["heic_support_active"]
        )

        if img_tensor is None:
            raise HTTPException(status_code=400, detail=f"Failed to preprocess image: {original_filename}")

        img_tensor = img_tensor.unsqueeze(0).to(app_state["device"])

        with torch.no_grad():
            outputs_dict = app_state["inference_model"](img_tensor)

        predicted_labels_details = {}
        meets_criteria = True # Cờ để kiểm tra tiêu chí confidence

        for task_name, logits in outputs_dict.items():
            probabilities = F.softmax(logits, dim=1)
            confidence_tensor, predicted_idx_tensor = torch.max(probabilities, 1)
            idx_val = predicted_idx_tensor.cpu().item()
            conf_val = confidence_tensor.cpu().item() # Giá trị confidence dạng float (0-1)

            label_str = f"UNKNOWN_IDX({idx_val})" # Giá trị mặc định

            if task_name not in label_encoder_map_mtl:
                label_str = f"NO_ENCODER_IDX({idx_val})"
                print(f"WARN: LabelEncoder for task '{task_name}' not found. Displaying index.")
                # Nếu task quan trọng thiếu encoder, đánh dấu không đạt
                if task_name in ['category', 'sub_category', 'orientation']:
                    meets_criteria = False
            else:
                try:
                    label_str = label_encoder_map_mtl[task_name].inverse_transform([idx_val])[0]
                except Exception as e_le:
                    label_str = f"LABEL_ERR_IDX({idx_val})"
                    print(f"Error decoding label for task {task_name}, index {idx_val}: {e_le}")
                    # Nếu lỗi decode label của task quan trọng, đánh dấu không đạt
                    if task_name in ['category', 'sub_category', 'orientation']:
                        meets_criteria = False

            # === KIỂM TRA CONFIDENCE ===
            # Chỉ kiểm tra cho 'category', 'sub_category', 'orientation'
            if task_name in ['category', 'sub_category', 'orientation']:
                if (conf_val * 100) < 90.0:
                    meets_criteria = False
                    print(f"INFO: Task '{task_name}' failed confidence check: {conf_val*100:.1f}% < 90%")
            # 'level' không cần kiểm tra

            predicted_labels_details[task_name] = PredictionDetail(
                label=label_str, confidence=f"{conf_val*100:.1f}%"
            )

        # === KIỂM TRA CỜ ===
        if not meets_criteria:
            # Nếu có bất kỳ task quan trọng nào không đạt, trả về lỗi 400
            print(f"INFO: Image '{original_filename}' did not meet the 90% confidence criteria.")
            raise HTTPException(status_code=400, detail="Ảnh không phù hợp với tiêu chí (confidence < 90%).")

        # Nếu tất cả đều đạt, trả về kết quả bình thường
        return PredictionResponse(filename=original_filename, predictions=predicted_labels_details)

    except HTTPException: # Re-throw HTTPException để FastAPI xử lý
        raise
    except Exception as e:
        print(f"ERROR: Unexpected error during prediction for {original_filename}: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Internal server error processing the image.")
    finally:
        if os.path.exists(temp_image_path):
            try:
                os.remove(temp_image_path)
            except Exception as e_remove:
                print(f"WARN: Could not remove temporary file {temp_image_path}: {e_remove}")

# ==== MAIN (Chỉ để chạy local) ==== (Không thay đổi)
if __name__ == "__main__":
    print("--- Starting Uvicorn server for LOCAL DEVELOPMENT ---")
    local_port = int(os.getenv("PORT", 8001))
    print(f"INFO: Model path for local dev: {MODEL_PATH_SERVER}")
    print(f"INFO: UPLOAD_DIR for local dev: {UPLOAD_DIR} (will be managed by lifespan)")
    print(f"INFO: Initial HEIC support status: {app_state['heic_support_active']}")

    uvicorn.run(app, host="0.0.0.0", port=local_port, log_level="info")
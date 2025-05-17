#!/usr/bin/env python3
import os
import io
import uuid # Để tạo tên file tạm thời duy nhất
from typing import Dict, Optional
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
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
    HEIC_SUPPORT_PILLOW = True
    print("INFO: pillow_heif imported and HEIC opener registered successfully.")
except ImportError:
    HEIC_SUPPORT_PILLOW = False
    print("WARN: pillow_heif library not found. HEIC file support will be unavailable.")
except Exception as e_heif_reg:
    HEIC_SUPPORT_PILLOW = True # Giả sử import thành công nhưng có vấn đề với register_opener
    print(f"INFO: pillow_heif imported. Note on opener registration: {e_heif_reg}")

# --- FastAPI và các thành phần liên quan ---
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn # Giữ lại uvicorn để chạy local


# ==== A. ĐỊNH NGHĨA CÁC THÀNH PHẦN CẦN THIẾT ====

# 1. Định nghĩa Class MultiTaskModel
class MultiTaskModel(nn.Module):
    """Model Multi-Task với backbone và các head riêng cho từng task."""
    def __init__(self, backbone_name="efficientnet_b0", pretrained=True, num_classes_per_task_dict=None):
        super().__init__()
        self.backbone_name = backbone_name
        self.num_classes_per_task_dict = num_classes_per_task_dict
        if num_classes_per_task_dict is None or not num_classes_per_task_dict:
            raise ValueError("num_classes_per_task_dict không được rỗng và phải được cung cấp.")

        self.task_names = list(num_classes_per_task_dict.keys())
        num_backbone_features = 0
        weights_eff = None
        weights_res = None
        if pretrained:
            if backbone_name == "efficientnet_b0":
                weights_eff = EfficientNet_B0_Weights.IMAGENET1K_V1
            elif backbone_name == "resnet18":
                weights_res = ResNet18_Weights.IMAGENET1K_V1

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
                head_level = nn.Sequential(
                    nn.Linear(num_backbone_features, num_backbone_features // 2),
                    nn.BatchNorm1d(num_backbone_features // 2),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(num_backbone_features // 2, num_classes)
                )
                self.heads[task_name] = head_level
            elif task_name == "sub_category":
                head_subcat = nn.Sequential(
                    nn.Linear(num_backbone_features, num_backbone_features // 4),
                    nn.ReLU(),
                    nn.Dropout(0.25),
                    nn.Linear(num_backbone_features // 4, num_classes)
                )
                self.heads[task_name] = head_subcat
            else: # For 'category' and 'orientation'
                self.heads[task_name] = nn.Linear(num_backbone_features, num_classes)
        if not self.heads:
            raise ValueError("Không có head nào được khởi tạo cho model.")

    def forward(self, x):
        features = self.backbone(x)
        if features.ndim == 4:
            features = features.view(features.size(0), -1)
        outputs = {task_name: head(features) for task_name, head in self.heads.items()}
        return outputs

# 2. Định nghĩa hàm full_process_image
def full_process_image(original_image_path, target_size_tuple=(224, 224), use_pillow_heif_flag=True):
    try:
        img = None
        file_ext = os.path.splitext(original_image_path)[1].lower()

        if use_pillow_heif_flag and (file_ext == ".heic" or file_ext == ".heif"):
            if not HEIC_SUPPORT_PILLOW:
                print(f"ERROR [full_process_image]: pillow_heif is not available, cannot process HEIC file {original_image_path}.")
                return None
            try:
                # Đảm bảo pillow_heif đã được đăng ký đúng cách
                # pillow_heif.register_heif_opener() # Có thể gọi lại ở đây nếu cần, nhưng thường gọi 1 lần là đủ
                img = Image.open(original_image_path) # Pillow sẽ tự dùng HEIF opener nếu đã đăng ký
            except Exception as e_heic:
                print(f"ERROR [full_process_image]: Failed to read HEIC {original_image_path} with Pillow/pillow_heif: {e_heic}.")
                return None
        else:
            img = Image.open(original_image_path)

        img = img.convert("RGB") # Đảm bảo ảnh là RGB
        img = img.resize(target_size_tuple, Image.Resampling.BILINEAR)
        return np.array(img)
    except FileNotFoundError:
        print(f"ERROR [full_process_image]: File not found: {original_image_path}")
        return None
    except Exception as e:
        print(f"ERROR [full_process_image]: Could not process {original_image_path}. Error: {e}")
        return None

# 3. Định nghĩa `config`
# Đường dẫn model khi deploy lên Render (model nằm ở thư mục gốc của project)
MODEL_PATH_SERVER   = os.getenv("MODEL_PATH", "./best_multitask_model_generic_last.pth")
UPLOAD_DIR          = "/tmp/image_uploads" # Thư mục tạm để lưu ảnh upload trên Render

# Đảm bảo thư mục upload tồn tại (an toàn khi chạy ở nhiều môi trường)
# Render cung cấp /tmp có thể ghi
if os.access("/tmp", os.W_OK) and not os.path.exists(UPLOAD_DIR):
    try:
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        print(f"INFO: Created UPLOAD_DIR at {UPLOAD_DIR}")
    except Exception as e:
        print(f"WARN: Could not create UPLOAD_DIR {UPLOAD_DIR}. Error: {e}")
elif os.path.exists(UPLOAD_DIR):
    print(f"INFO: UPLOAD_DIR {UPLOAD_DIR} already exists.")
else:
    print(f"WARN: Cannot write to /tmp, UPLOAD_DIR {UPLOAD_DIR} may not be available or writable.")


config = {
    'img_size': 224,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu', # Render sẽ dùng CPU
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

# 4. Định nghĩa `label_encoder_map_mtl`
label_encoder_map_mtl = {}
le_category = LabelEncoder(); le_category.fit(['cruve', 'straight']); label_encoder_map_mtl['category'] = le_category
le_sub_category = LabelEncoder(); le_sub_category.fit(['circle', 'cruve', 'line', 'spiral', 'square', 'wave', 'zigzag']); label_encoder_map_mtl['sub_category'] = le_sub_category
le_level = LabelEncoder(); le_level.fit(['Level 1', 'Level 2', 'Level 3', 'Level 4', 'Level 5']); label_encoder_map_mtl['level'] = le_level
le_orientation = LabelEncoder(); le_orientation.fit(['back', 'front']); label_encoder_map_mtl['orientation'] = le_orientation

# --- B. CÁC HÀM TIỆN ÍCH ---

def load_trained_model_server(model_path, model_class_def, config_for_model, device_to_load):
    print(f"INFO: Attempting to load model from: {model_path}")
    # Kiểm tra xem model_path có phải là đường dẫn tuyệt đối không, nếu không thì tạo đường dẫn tuyệt đối từ thư mục gốc của project
    # Tuy nhiên, Gunicorn thường chạy từ thư mục gốc, nên "./model.pth" là OK
    if not os.path.isabs(model_path) and not model_path.startswith("./"):
         # Heuristic: if it's just a filename, assume it's in the project root
         if not os.path.exists(model_path) and os.path.exists(os.path.join(os.getcwd(), model_path)):
              model_path = os.path.join(os.getcwd(), model_path)
         elif not os.path.exists(model_path) and os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', model_path)): # Relative to app/main.py parent
              model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', model_path)


    model = model_class_def(
        backbone_name=config_for_model["mtl_model_name"],
        pretrained=config_for_model["mtl_pretrained_backbone"],
        num_classes_per_task_dict=config_for_model["num_classes_per_task"]
    )
    try:
        if not os.path.exists(model_path):
            print(f"ERROR: Model file not found at {model_path}. Current PWD: {os.getcwd()}")
            return None
        print(f"INFO: Loading model state_dict from validated path: {model_path}")
        state_dict = torch.load(model_path, map_location=torch.device(device_to_load))
        model.load_state_dict(state_dict, strict=True)
        print("INFO: Successfully loaded model weights (strict=True).")
    except RuntimeError as e_strict_true:
        print(f"ERROR: Failed to load model weights with strict=True. Error: {e_strict_true}")
        return None
    except Exception as e_load:
        print(f"ERROR: An unexpected error occurred while loading the model: {e_load}")
        return None
    model.to(device_to_load)
    model.eval()
    return model

def preprocess_single_image_for_mtl_server(original_image_path, target_size, ptorch_transforms, use_pillow_heif_flag):
    if ptorch_transforms is None:
        print("ERROR [preprocess_single_image_for_mtl_server]: PyTorch transforms missing.")
        return None
    processed_np_array = full_process_image(
        original_image_path, target_size_tuple=(target_size, target_size),
        use_pillow_heif_flag=use_pillow_heif_flag
    )
    if processed_np_array is None: return None
    try:
        pil_image = Image.fromarray(processed_np_array)
        img_tensor = ptorch_transforms(pil_image)
        return img_tensor
    except Exception as e_tf:
        print(f"ERROR [preprocess_single_image_for_mtl_server]: Image to Tensor or PyTorch transform failed: {e_tf}")
        return None

# ==== GLOBAL VARIABLES FOR SERVER ====
inference_model = None
inference_transforms_global = None
DEVICE_SERVER = config['device'] # Sẽ là 'cpu' trên Render free tier

# ==== LIFESPAN EVENTS ====
@asynccontextmanager
async def lifespan(app_lifespan: FastAPI): # Đổi tên tham số để tránh trùng với 'app' global
    # Startup
    global inference_model, inference_transforms_global, DEVICE_SERVER
    print(f"INFO: Server starting up. Using device: {DEVICE_SERVER}")
    print(f"INFO: HEIC Support via Pillow-HEIF: {HEIC_SUPPORT_PILLOW}")

    try:
        inference_transforms_global = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    except Exception as e_init_transforms:
        print(f"CRITICAL ERROR: Could not create inference_transforms: {e_init_transforms}")
        pass

    print(f"Attempting to load model. Expected path relative to PWD: {MODEL_PATH_SERVER}")
    inference_model = load_trained_model_server(MODEL_PATH_SERVER, MultiTaskModel, config, DEVICE_SERVER)
    if inference_model is None:
        print("CRITICAL ERROR: Failed to load the model. The /predict endpoint will not function.")
    else:
        print("INFO: Model loaded successfully. Server is ready to accept requests for prediction.")
    
    print("INFO: Application startup actions complete.")
    yield
    # Shutdown
    print("INFO: Server shutting down.")

# ==== FASTAPI APP ====
# Đây là instance mà Gunicorn sẽ tìm (app.main:app)
app = FastAPI(
    title="Multi-Task Image Classification Server (Render)",
    description="Upload an image to get classifications for category, sub_category, level, and orientation.",
    version="1.0",
    lifespan=lifespan # Sử dụng lifespan manager
)

# ==== CORS MIDDLEWARE ====
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Cho phép tất cả origins, hoặc chỉ định domain cụ thể
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==== Pydantic Model cho response body ====
class PredictionDetail(BaseModel):
    label: str
    confidence: str

class PredictionResponse(BaseModel):
    filename: str
    predictions: Dict[str, PredictionDetail]
    error: Optional[str] = None

# ==== Health Check ====
@app.get("/health") # Render có thể dùng endpoint này
async def health():
    model_status = "loaded" if inference_model else "not loaded or failed to load"
    transforms_status = "initialized" if inference_transforms_global else "not initialized"
    return {
        "status": "ok" if inference_model and inference_transforms_global else "degraded",
        "model_status": model_status,
        "transforms_status": transforms_status,
        "device": DEVICE_SERVER,
        "heic_support": HEIC_SUPPORT_PILLOW
    }

# ==== Prediction Endpoint ====
@app.post("/predict", response_model=PredictionResponse)
async def predict_image(file: UploadFile = File(...)):
    if inference_model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded or failed to load. Cannot process predictions.")
    if inference_transforms_global is None:
        raise HTTPException(status_code=503, detail="Image transforms are not initialized. Cannot process predictions.")

    original_filename = file.filename
    file_extension = os.path.splitext(original_filename)[1].lower()
    # Tạo tên file tạm thời duy nhất trong UPLOAD_DIR
    temp_filename = f"{uuid.uuid4()}{file_extension}" # Đảm bảo UPLOAD_DIR đã được tạo
    
    # Kiểm tra lại UPLOAD_DIR phòng trường hợp không tạo được ở global scope
    if not os.path.exists(UPLOAD_DIR) and os.access("/tmp", os.W_OK) :
        try:
            os.makedirs(UPLOAD_DIR, exist_ok=True)
        except Exception as e_mkdir:
             raise HTTPException(status_code=500, detail=f"Could not create temp upload directory: {e_mkdir}")
    elif not os.access(UPLOAD_DIR, os.W_OK):
        raise HTTPException(status_code=500, detail=f"Temp upload directory {UPLOAD_DIR} is not writable.")

    temp_image_path = os.path.join(UPLOAD_DIR, temp_filename)


    try:
        with open(temp_image_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        if (file_extension == ".heic" or file_extension == ".heif") and not HEIC_SUPPORT_PILLOW:
            msg = f"Cannot process HEIC/HEIF file '{original_filename}'. Server lacks HEIC support (pillow_heif not available/working)."
            print(f"ERROR: {msg}")
            raise HTTPException(status_code=400, detail=msg)

        img_tensor = preprocess_single_image_for_mtl_server(
            temp_image_path,
            config['img_size'],
            inference_transforms_global,
            use_pillow_heif_flag=HEIC_SUPPORT_PILLOW
        )

        if img_tensor is None:
            print(f"ERROR: Could not preprocess image {original_filename}.")
            raise HTTPException(status_code=400, detail=f"Failed to preprocess image: {original_filename}")

        img_tensor = img_tensor.unsqueeze(0).to(DEVICE_SERVER)

        with torch.no_grad():
            outputs_dict = inference_model(img_tensor)

        predicted_labels_details = {}
        for task_name, logits in outputs_dict.items():
            if task_name not in label_encoder_map_mtl:
                idx_val = torch.max(logits, 1)[1].cpu().item()
                conf_val = torch.max(F.softmax(logits, dim=1), 1)[0].cpu().item()
                predicted_labels_details[task_name] = PredictionDetail(
                    label=f"NO_ENCODER_IDX({idx_val})",
                    confidence=f"{conf_val*100:.1f}%"
                )
                print(f"WARN: LabelEncoder for task '{task_name}' not found. Displaying index.")
                continue

            probabilities = F.softmax(logits, dim=1)
            confidence_tensor, predicted_idx_tensor = torch.max(probabilities, 1)
            idx_val = predicted_idx_tensor.cpu().item()
            conf_val = confidence_tensor.cpu().item()
            try:
                label_str = label_encoder_map_mtl[task_name].inverse_transform([idx_val])[0]
            except Exception as e_le:
                label_str = f"LABEL_ERR_IDX({idx_val})"
                print(f"Error decoding label for task {task_name}, index {idx_val}: {e_le}")

            predicted_labels_details[task_name] = PredictionDetail(
                label=label_str,
                confidence=f"{conf_val*100:.1f}%"
            )
        return PredictionResponse(filename=original_filename, predictions=predicted_labels_details)

    except HTTPException as e: # Re-throw HTTPException để FastAPI xử lý
        raise e
    except Exception as e:
        print(f"ERROR: An unexpected error occurred during prediction for {original_filename}: {e}")
        # Ghi log chi tiết hơn về lỗi
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal server error processing the image.")
    finally:
        # Xóa file tạm sau khi xử lý
        if os.path.exists(temp_image_path):
            try:
                os.remove(temp_image_path)
            except Exception as e_remove:
                print(f"WARN: Could not remove temporary file {temp_image_path}: {e_remove}")


# ==== MAIN (Chỉ để chạy local, Render sẽ không dùng khối này) ====
if __name__ == "__main__":
    print("--- Starting Uvicorn server for LOCAL DEVELOPMENT ---")
    # PORT được Render inject, nhưng ở local bạn có thể set cố định
    local_port = int(os.getenv("PORT", 8001))
    print(f"INFO: Model path for local dev: {MODEL_PATH_SERVER}")
    print(f"INFO: UPLOAD_DIR for local dev: {UPLOAD_DIR}")
    print(f"INFO: HEIC support for local dev: {HEIC_SUPPORT_PILLOW}")

    # Kiểm tra xem model có load được không khi chạy local
    if inference_model is None and lifespan: # lifespan chỉ được gọi bởi FastAPI khi chạy với uvicorn
        print("WARN: Model not loaded yet. Uvicorn will trigger lifespan to load it.")
    elif inference_model:
        print("INFO: Model seems to be pre-loaded (possibly due to import-time call to lifespan or previous run).")


    uvicorn.run(app, host="0.0.0.0", port=local_port, log_level="info")
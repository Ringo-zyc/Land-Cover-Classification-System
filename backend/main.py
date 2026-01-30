from cyclegan.models.networks import define_G
import requests
import os
import uuid
import sys
import cv2
import torch
import numpy as np
from PIL import Image
import albumentations as albu
import logging
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Request, Query
from fastapi.security import OAuth2PasswordBearer
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy.orm import Session
from passlib.context import CryptContext
from jose import JWTError, jwt
from datetime import datetime, timedelta
import io
import shutil
from typing import List
import asyncio
from concurrent.futures import ThreadPoolExecutor
import aiofiles
from database import SessionLocal, engine, Base, get_db
from db_models import User, SegmentationHistory
from schemas import UserCreate, UserLogin

# 路径配置
current_dir = os.path.dirname(os.path.abspath(__file__))
IMAGE_STORAGE = os.path.join(current_dir, 'images')
SEGMENTED_STORAGE = os.path.join(current_dir, 'segmented_images')
STYLE_TRANSFER_STORAGE = os.path.join(current_dir, 'style_transfer_images')
os.makedirs(IMAGE_STORAGE, exist_ok=True)
os.makedirs(SEGMENTED_STORAGE, exist_ok=True)
os.makedirs(STYLE_TRANSFER_STORAGE, exist_ok=True)

# 初始化数据库
Base.metadata.create_all(bind=engine)

app = FastAPI()

# CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8081", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局异常处理器
@app.exception_handler(Exception)
async def exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal Server Error"},
        headers={"Access-Control-Allow-Origin": "*"}
    )

# 挂载静态文件目录
app.mount("/images", StaticFiles(directory=IMAGE_STORAGE), name="images")
app.mount("/segmented_images", StaticFiles(directory=SEGMENTED_STORAGE), name="segmented_images")
app.mount("/style_transfer_images", StaticFiles(directory=STYLE_TRANSFER_STORAGE), name="style_transfer_images")

# 日志配置
logging.basicConfig(
    filename='app.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 导入模型
sys.path.append(os.path.join(current_dir, 'unetmamba_model'))
try:
    from unetmamba_model.models.UNetMamba import UNetMamba
    from unetmamba_model.models.UNetFormer import UNetFormer
    from unetmamba_model.models.DCSwin import dcswin_small
    logger.info("Models imported successfully")
except ImportError as e:
    logger.error(f"Failed to import models: {e}")
    raise
    



# 统一的 PALETTE
PALETTE = [
    [255, 255, 255],  # Background
    [255, 0, 0],      # Building
    [255, 255, 0],    # Road
    [0, 0, 255],      # Water
    [159, 129, 183],  # Barren
    [0, 255, 0],      # Forest
    [255, 195, 128]   # Agriculture
]

CLASS_NAMES = [
    'Background',
    'Building',
    'Road',
    'Water',
    'Barren',
    'Forest',
    'Agriculture'
]

CLASS_NAME_TO_COLOR = {
    'Background': [255, 255, 255],
    'Building':   [255, 0, 0],
    'Road':       [255, 255, 0],
    'Water':      [0, 0, 255],
    'Barren':     [159, 129, 183],
    'Forest':     [0, 255, 0],
    'Agriculture':[255, 195, 128]
}

MODEL_CLASS_INDEX_MAP = {
    "unetmamba": { #  unetmamba 的映射关系 (作为参考)
        0: "Background",
        1: "Building",
        2: "Road",
        3: "Water",
        4: "Barren",
        5: "Forest",
        6: "Agriculture"
    },
    "dcswin_small": { #  **根据您的日志和推测进行调整，需要验证**
        0: "Barren",
        1: "Road",
        2: "Building",
        3: "Water",
        5: "Background",
        6: "Forest",
        4: "Agriculture"
    },
    "unetformer_r18": { # **尝试将 UNetformer_r18 映射到 UNetMamba 的颜色，** **修改索引 0 为 'none'**
        0: "none",     # **将索引 0 映射为 'none'**
        1: "Building",
        2: "Road",
        3: "Water",
        4: "Agriculture",
        5: "Forest",
        6: "Barren"
    },
}





# 模型注册字典
MODEL_REGISTRY = {
    "unetmamba": {
        "class": UNetMamba,
        "weights_path": os.path.join(current_dir, "model_weights/loveda/unetmamba-1024-e100-v1.ckpt"),
        "args": {
            "pretrained": None,
            "num_classes": 7,
            "patch_size": 4,
            "in_chans": 3,
            "depths": [2, 2, 9, 2],
            "dims": 96,
            "ssm_d_state": 16,
            "ssm_ratio": 2.0,
            "ssm_rank_ratio": 2.0,
            "ssm_dt_rank": "auto",
            "ssm_act_layer": "silu",
            "ssm_conv": 3,
            "ssm_conv_bias": True,
            "ssm_drop_rate": 0.0,
            "ssm_init": "v0",
            "forward_type": "v4",
            "mlp_ratio": 4.0,
            "mlp_act_layer": "gelu",
            "mlp_drop_rate": 0.0,
            "drop_path_rate": 0.1,
            "patch_norm": True,
            "norm_layer": "ln",
            "downsample_version": "v2",
            "patchembed_version": "v2",
            "gmlp": False,
            "use_checkpoint": False
        }
    },
    "dcswin_small": {
        "factory_func": dcswin_small,
        "args": {
            "num_classes": 7,
            "pretrained": False,
            "weight_path": os.path.join(current_dir, "pretrain_weights/stseg_small.pth")
        },
        "weights_path": os.path.join(current_dir, "model_weights/loveda/dcswin-small-512crop-ms-epoch30.ckpt")
    },
    "unetformer_r18": {
        "class": UNetFormer,
        "args": {
            "num_classes": 7,
            "decode_channels": 64,
            "dropout": 0.1,
            "backbone_name": "swsl_resnet18",
            "pretrained": False,
            "window_size": 8
        },
        "weights_path": os.path.join(current_dir, "model_weights/loveda/unetformer-r18-512crop-ms-epoch30-rep.ckpt")
    },
     "cyclegan": {
        "factory_func": define_G,
        "args": {
            "input_nc": 3,      #   <---  **添加 input_nc 位置参数**
            "output_nc": 3,     #   <---  **添加 output_nc 位置参数**
            "ngf": 64,          #   <---  **添加 ngf 位置参数**
            "netG": 'resnet_9blocks',
            "norm": 'instance',
            "init_type": 'normal',
            "init_gain": 0.02,
            "gpu_ids": [],
        },
        "weights_path": os.path.join(current_dir, "cyclegan", "checkpoints", "latest_net_G_A.pth")
    }
}


# 密码哈希和 JWT 配置
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")
load_dotenv()
SECRET_KEY = os.getenv("SECRET_KEY")
logger.info(f"Loaded SECRET_KEY: {SECRET_KEY}")
ALGORITHM = "HS256"

# 全局模型存储
models = {}

# 获取数据库会话
def get_db():
    db = SessionLocal()
    try:
        yield db
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()

# 生成 JWT token
def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(hours=1)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

# 获取当前用户
async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid token")
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")
    user = db.query(User).filter(User.username == username).first()
    if user is None:
        raise HTTPException(status_code=401, detail="User not found")
    return user

# 模型加载函数
def load_model(model_name: str):
    if model_name not in MODEL_REGISTRY:
        raise HTTPException(status_code=400, detail=f"Invalid model name: {model_name}")

    model_info = MODEL_REGISTRY[model_name]
    if "factory_func" in model_info:
        model = model_info["factory_func"](**model_info["args"])
        weights_path = model_info.get("weights_path")
        if weights_path and os.path.exists(weights_path):
            checkpoint = torch.load(weights_path, map_location="cpu")
            state_dict = checkpoint.get("state_dict", checkpoint)
            model.load_state_dict(state_dict, strict=False)
    else:
        model_class = model_info["class"]
        model_args = model_info.get("args", {})
        model = model_class(**model_args)
        weights_path = model_info["weights_path"]
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Model weights file not found: {weights_path}")
        checkpoint = torch.load(weights_path, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint)
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('net.'):
                new_key = key[4:]
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value
        model.load_state_dict(new_state_dict, strict=True)

    model.eval()
    if torch.cuda.is_available():
        model.cuda()
    logger.info(f"Model {model_name} loaded successfully")
    return model

# 图像预处理
def preprocess_image(image: Image.Image, model_name: str) -> torch.Tensor:
    if model_name == "unetmamba":
        size = (1024, 1024)
    elif model_name in ["dcswin_small", "unetformer_r18"]:
        size = (512, 512)
    elif model_name == "cyclegan": #  <--  针对 cyclegan 模型
        size = (256, 256) # CycleGAN 常用尺寸，可以根据您的模型调整
    else:
        size = (1024, 1024)
    image = image.resize(size)
    image = np.array(image)
    if model_name == "cyclegan": #  <--  CycleGAN 的预处理
        image = image / 127.5 - 1.0  # 像素值范围缩放到 [-1, 1]
        image = torch.from_numpy(image).permute(2, 0, 1).float().unsqueeze(0) # 转换为 Tensor
    else: #  其他模型使用之前的归一化方式
        transform = albu.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        image = transform(image=image)["image"]
        image = torch.from_numpy(image).permute(2, 0, 1).float().unsqueeze(0)
    if torch.cuda.is_available():
        image = image.cuda()
    return image

# 图像后处理
def postprocess_output(output: torch.Tensor, model_name: str) -> np.ndarray:
    """
    处理模型输出，生成 RGB 格式的分割掩码, **并应用颜色映射**。

    Args:
        output (torch.Tensor): 模型输出张量
        model_name (str): 使用的模型名称

    Returns:
        np.ndarray: RGB 格式的分割掩码
    """
    if model_name == "cyclegan":
        # CycleGAN 输出处理 (保持不变)
        output = output.squeeze().cpu().detach()
        output = (output + 1) / 2
        output = output.permute(1, 2, 0).numpy()
        output = (output * 255).astype(np.uint8)
        return output
    else:
        # 分割模型的处理 (颜色映射逻辑整合到这里)
        output = torch.softmax(output, dim=1)
        mask = output.argmax(dim=1).squeeze().cpu().numpy()
        h, w = mask.shape
        mask_rgb = np.zeros((h, w, 3), dtype=np.uint8)

        # 统计 Mask 中每个索引值的像素数量 (日志记录 - 可选)
        unique_indices, counts = np.unique(mask, return_counts=True)
        index_counts = dict(zip(unique_indices, counts))
        logger.info(f"Model: {model_name}, Index Counts: {index_counts}")

        # 获取模型特定的类别映射
        model_index_map = MODEL_CLASS_INDEX_MAP.get(model_name)
        if not model_index_map:
            raise ValueError(f"未知的模型名称: {model_name}")

        # **应用颜色映射**
        for i in range(h):
            for j in range(w):
                index = mask[i, j]
                class_name = model_index_map.get(index, 'Background') #  如果索引不在映射中，默认映射到 Background
                color = CLASS_NAME_TO_COLOR[class_name]
                mask_rgb[i, j] = color

        logger.info(f"Model: {model_name}, mask_rgb Shape: {mask_rgb.shape}")
        return mask_rgb


    
@app.post("/api/style_transfer")
async def style_transfer(file: UploadFile = File(...), style_type: str = "default", current_user: User = Depends(get_current_user)):
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="上传的文件不是图像")

    # 加载CycleGAN模型
    model_name = "cyclegan"
    if model_name not in models:
        models[model_name] = load_model(model_name)
    model = models[model_name]

    # 读取上传的图像
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data)).convert('RGB')
    input_tensor = preprocess_image(image, model_name)

    # 模型推理
    with torch.no_grad():
        output = model(input_tensor)

    # 后处理
    styled_image = postprocess_output(output, model_name)
    styled_image_pil = Image.fromarray(styled_image)

    # 保存风格转换图像
    image_id = str(uuid.uuid4())
    styled_filename = f"{image_id}_styled.png"
    styled_path = os.path.join(STYLE_TRANSFER_STORAGE, styled_filename)
    styled_image_pil.save(styled_path)

    styled_url = f"/style_transfer_images/{styled_filename}"
    return JSONResponse(content={"styledImageUrl": styled_url})
    

# DeepSeek API Key
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
if not DEEPSEEK_API_KEY:
    print("请设置 DEEPSEEK_API_KEY 环境变量")

# 图像分析报告生成接口
@app.post("/api/generate_report")
async def generate_report(request: Request, current_user: User = Depends(get_current_user)):
    data = await request.json()
    prompt = data.get('prompt')
    segmentation_stats = data.get('segmentationStats', {})  # 获取前端传递的统计数据，默认值为空字典

    if not prompt:
        raise HTTPException(status_code=400, detail="缺少 Prompt")

    # 将 segmentationStats 格式化为字符串
    stats_str = "\n".join([f"{key}: {value}" for key, value in segmentation_stats.items()])
    # 构建完整的提示词
    full_prompt = f"{prompt}\n\n分割统计数据:\n{stats_str}"

    try:
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {DEEPSEEK_API_KEY}',
        }
        loop = asyncio.get_running_loop()
        deepseek_response = await loop.run_in_executor(
            None,
            lambda: requests.post(
                "https://api.deepseek.com/v1/chat/completions",
                headers=headers,
                json={
                    "model": "deepseek-chat",
                    "messages": [
                        {"role": "system", "content": "你是一位遥感图像分析专家。"},
                        {"role": "user", "content": full_prompt}  # 使用完整的提示词
                    ],
                    "max_tokens": 1500,
                    "temperature": 0.7
                }
            )
        )
        deepseek_response.raise_for_status()
        report_text = deepseek_response.json()['choices'][0]['message']['content'].strip()

        report_filename = f"report_{uuid.uuid4()}.txt"
        report_filepath = os.path.join(SEGMENTED_STORAGE, report_filename)
        async with aiofiles.open(report_filepath, "w", encoding="utf-8") as f:
            await f.write(report_text)

        report_url = f"/segmented_images/{report_filename}"
        return JSONResponse(content={"report": report_text, "reportUrl": report_url})
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"报告生成失败，DeepSeek API 调用出错: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"报告生成失败，请稍后重试: {e}")
        

# 风格化图像生成接口
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("请设置 OPENAI_API_KEY 环境变量")

class StyledImageRequest(BaseModel):
    stylePrompt: str
    segmentationStats: dict  # 分割统计数据，例如 {"Building": 40, "Forest": 25, "Water": 15}

@app.post("/api/generate_styled_image")
async def generate_styled_image(request: StyledImageRequest, current_user: str = Depends(get_current_user)):
    """
    根据提供的风格提示词和分割统计数据生成风格化图像。
    """
    if not current_user:
        raise HTTPException(status_code=401, detail="未授权的用户")

    if not request.stylePrompt:
        raise HTTPException(status_code=400, detail="风格提示词不能为空")

    if not request.segmentationStats:
        raise HTTPException(status_code=400, detail="分割统计数据不能为空")

    # 从分割统计数据中提取主要类别
    sorted_stats = sorted(request.segmentationStats.items(), key=lambda item: item[1], reverse=True)
    top_classes = ", ".join([f"{label}" for label, percentage in sorted_stats[:3]])  # 取前 3 个主要类别

    # 构建完整 Prompt，包含风格提示词和图像内容描述
    full_prompt = f"Generate a stylized image in {request.stylePrompt} style, depicting remote sensing image with dominant land cover classes: {top_classes}."

    print(f"Prompt sent to OpenAI: {full_prompt}")  # 打印发送给 OpenAI 的 Prompt

    try:
        # 设置请求头
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        }

        # 使用代理服务提供的 URL 并保留 OpenAI 的图像生成端点
        url = "https://db04599540657cc539b72ed32725b26e.api-forwards.com/v1/images/generations"
        print(f"Requesting URL: {url}")  # 打印请求的 URL 以便调试

        # 发送 POST 请求
        response = requests.post(
            url,
            headers=headers,
            json={
                "prompt": full_prompt,
                "n": 1,
                "size": "1024x1024"
            },
            verify=False  # 禁用 SSL 证书验证，仅用于测试
        )

        if response.status_code == 200:
            styled_image_url = response.json()["data"][0]["url"]
            print(f"Generated image URL: {styled_image_url}")  # 打印 URL
            return {"styledImageUrl": styled_image_url}
        else:
            print(f"Error response: {response.text}")  # 打印错误响应以便调试
            raise HTTPException(status_code=500, detail=f"请求失败: {response.text}")

    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"OpenAI API 错误: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"图像生成失败: {str(e)}")
        
def construct_prompt(user_prompt_goal, image_description, segmentation_stats):
    """构建发送给 DeepSeek 的 Prompt"""
    prompt_template = """
    遥感图像分析报告

    图像描述： {image_description}

    分析目标： {user_prompt_goal}

    分割结果统计数据：
    {segmentation_stats_text}

    请基于以上信息，生成一份详细的遥感图像分析报告，报告应结构清晰、重点突出，并尽可能提供有价值的分析和解读。
    """

    segmentation_stats_text = ""
    if segmentation_stats:
        if isinstance(segmentation_stats, dict):
            for category, stats in segmentation_stats.items():
                if isinstance(stats, dict) and 'pixel_count' in stats and 'percentage' in stats:
                    segmentation_stats_text += f"- {category}: 像素数量={stats['pixel_count']}, 面积占比={stats['percentage']:.2f}%\n"
                else:
                    logger.warning(f"Unexpected stats format for category {category}: {stats}")
                    segmentation_stats_text += f"- {category}: 统计数据格式不正确\n"
        else:
            logger.warning(f"segmentation_stats is not a dictionary, but: {type(segmentation_stats)}, value: {segmentation_stats}")
            segmentation_stats_text = "分割结果统计数据格式不正确"
    else:
        segmentation_stats_text = "暂无分割结果统计数据"

    prompt = prompt_template.format(
        image_description=image_description if image_description else "未提供图像描述",
        user_prompt_goal=user_prompt_goal,
        segmentation_stats_text=segmentation_stats_text
    )
    return prompt

# 注册接口
@app.post("/api/register")
def register(user: UserCreate, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.username == user.username).first()
    if db_user:
        raise HTTPException(status_code=400, detail="用户名已存在")
    hashed_password = pwd_context.hash(user.password)
    new_user = User(username=user.username, password_hash=hashed_password)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return {"message": "注册成功"}

# 登录接口
@app.post("/api/login")
def login(user: UserLogin, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.username == user.username).first()
    if not db_user or not pwd_context.verify(user.password, db_user.password_hash):
        raise HTTPException(status_code=400, detail="用户名或密码错误")
    access_token = create_access_token(data={"sub": user.username})
    return {"access_token": access_token, "token_type": "bearer"}

# 上传图像接口（异步文件操作）
@app.post("/upload")
async def upload_images(files: List[UploadFile] = File(...)):
    if len(files) > 4:
        raise HTTPException(status_code=400, detail="最多允许上传4张图像")
    image_ids = []
    for file in files:
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail='上传的文件不是图像')
        image_id = str(uuid.uuid4())
        original_filename = f"{image_id}_original.png"
        file_path = os.path.join(IMAGE_STORAGE, original_filename)
        async with aiofiles.open(file_path, "wb") as buffer:
            content = await file.read()  # 异步读取文件内容
            await buffer.write(content)  # 异步写入
        image_ids.append(image_id)
    return JSONResponse(content={"imageIds": image_ids, "message": "上传成功"})

# 图像分割接口（异步模型推理 + 批量数据库操作）
executor = ThreadPoolExecutor(max_workers=4)  # 线程池，控制并发推理

# segment 接口
@app.post("/segment")
async def segment_image(data: dict, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """
    分割图像并返回结果。
    
    Args:
        data (dict): 请求数据，包含 imageIds, modelName, opacities
        current_user (User): 当前用户
        db (Session): 数据库会话
    
    Returns:
        JSONResponse: 分割结果
    """
    image_ids = data.get("imageIds", [])
    model_name = data.get("modelName", "unetmamba")
    opacities = data.get("opacities", {})
    if not image_ids:
        raise HTTPException(status_code=400, detail="需要图像 IDs")

    # 按需加载模型
    if model_name not in models:
        models[model_name] = load_model(model_name)
    model = models[model_name]

    results = []
    history_entries = []  # 批量收集历史记录
    for image_id in image_ids:
        input_path = os.path.join(IMAGE_STORAGE, f"{image_id}_original.png")
        if not os.path.exists(input_path):
            results.append({"error": "未找到图像"})
            continue
        image = Image.open(input_path).convert('RGB')
        input_tensor = preprocess_image(image, model_name)

        # 异步执行模型推理
        loop = asyncio.get_event_loop()
        output = await loop.run_in_executor(executor, lambda: run_model(model, input_tensor))

        # 获取 mask_rgb
        mask_rgb = postprocess_output(output, model_name)

        # 生成带透明度的 RGBA 图像
        h, w, _ = mask_rgb.shape
        mask_rgba = np.zeros((h, w, 4), dtype=np.uint8)
        for class_name in CLASS_NAMES:
            color = CLASS_NAME_TO_COLOR[class_name]
            alpha = int(float(opacities.get(class_name, 1.0)) * 255)
            mask_rgba[np.all(mask_rgb == color, axis=-1)] = [*color, alpha]

        segmented_image = Image.fromarray(mask_rgba, 'RGBA')
        output_path = os.path.join(SEGMENTED_STORAGE, f"{image_id}_segmented.png")
        segmented_image.save(output_path)

        # 计算统计数据
        mask = output.argmax(dim=1).squeeze().cpu().numpy()
        stats = {class_name: float((mask == i).sum() / mask.size * 100) for i, class_name in enumerate(CLASS_NAMES)}
        segmented_url = f"/segmented_images/{image_id}_segmented.png"

        # 创建历史记录对象
        history_entry = SegmentationHistory(
            user_id=current_user.id,
            image_id=image_id,
            segmented_url=segmented_url,
            stats=stats
        )
        history_entries.append(history_entry)
        results.append({"imageUrl": segmented_url, "stats": stats})

    # 批量插入数据库
    db.add_all(history_entries)
    db.commit()
    return JSONResponse(content={"results": results})

# 辅助函数：线程池执行模型推理
def run_model(model, input_tensor):
    with torch.no_grad():
        return model(input_tensor)

# 获取历史记录
@app.get("/history")
async def get_history(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    page: int = Query(1, ge=1),
    per_page: int = Query(10, ge=1, le=100)
):
    offset = (page - 1) * per_page
    history_records = (
        db.query(SegmentationHistory)
        .filter(SegmentationHistory.user_id == current_user.id)
        .order_by(SegmentationHistory.created_at.desc())
        .offset(offset)
        .limit(per_page)
        .all()
    )
    total = db.query(SegmentationHistory).filter(SegmentationHistory.user_id == current_user.id).count()

    results = [
        {
            "image_id": record.image_id,
            "original_url": f"/images/{record.image_id}_original.png",
            "segmented_url": record.segmented_url,
            "stats": record.stats,
            "created_at": record.created_at.isoformat()
        }
        for record in history_records
    ]
    return {"history": results, "total": total, "page": page, "per_page": per_page}

# 清除历史记录
@app.delete("/clear_history")
async def clear_history(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    records = db.query(SegmentationHistory).filter(SegmentationHistory.user_id == current_user.id).all()
    for record in records:
        image_path = os.path.join(IMAGE_STORAGE, f"{record.image_id}_original.png")
        segmented_path = os.path.join(SEGMENTED_STORAGE, f"{record.image_id}_segmented.png")
        if os.path.exists(image_path):
            os.remove(image_path)
        if os.path.exists(segmented_path):
            os.remove(segmented_path)
    db.query(SegmentationHistory).filter(SegmentationHistory.user_id == current_user.id).delete()
    db.commit()
    return {"message": "历史记录和相关文件已清除"}

# 删除单条历史记录
@app.delete("/history/{image_id}")
async def delete_history_item(image_id: str, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    record = db.query(SegmentationHistory).filter(
        SegmentationHistory.image_id == image_id,
        SegmentationHistory.user_id == current_user.id
    ).first()
    if not record:
        raise HTTPException(status_code=404, detail="历史记录未找到")
    
    # 删除文件
    image_path = os.path.join(IMAGE_STORAGE, f"{record.image_id}_original.png")
    segmented_path = os.path.join(SEGMENTED_STORAGE, f"{record.image_id}_segmented.png")
    if os.path.exists(image_path):
        os.remove(image_path)
    if os.path.exists(segmented_path):
        os.remove(segmented_path)
    
    # 删除数据库记录
    db.delete(record)
    db.commit()
    return {"message": "历史记录已删除"}

# 下载分割后图像
@app.get("/download/{imageId}")
async def download_image(imageId: str):
    segmented_file_path = os.path.join(SEGMENTED_STORAGE, f"{imageId}_segmented.png")
    if not os.path.exists(segmented_file_path):
        raise HTTPException(status_code=404, detail="未找到分割图像")
    return FileResponse(segmented_file_path, media_type="image/png", filename="segmented_image.png")

# 下载分析报告接口
@app.get("/download_report")
async def download_report(reportUrl: str):
    """下载分析报告"""
    if not reportUrl:
        raise HTTPException(status_code=400, detail="需要报告 URL")

    report_file_path = os.path.join(current_dir, reportUrl.lstrip("/"))  # 构建本地文件路径
    if not os.path.exists(report_file_path):
        raise HTTPException(status_code=404, detail="未找到报告文件")

    return FileResponse(report_file_path, media_type="text/plain", filename=os.path.basename(report_file_path))

if __name__ == '__main__':
    import uvicorn
    import requests  # 导入 requests 库

    uvicorn.run(app, host='0.0.0.0', port=8000, workers=4)  # 启动时使用 4 个 worker
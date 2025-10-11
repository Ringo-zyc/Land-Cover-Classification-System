from fastapi import FastAPI, File, UploadFile
import io
from PIL import Image
import numpy as np
import torch
import model_loader  # 导入模型加载模块
import os

# 创建 FastAPI 实例
app = FastAPI()

# 确保 uploads 目录存在
UPLOAD_FOLDER = "D:/remote-sensing-segmentation-app/backend/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.get("/")
def root():
    return {"message": "欢迎使用遥感图像分割 API！请访问 /docs 测试 API。"}


@app.post("/upload/")
async def upload_image(file: UploadFile = File(...)):
    """
    上传图片并保存到服务器
    """
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())

    return {"filename": file.filename, "message": "上传成功"}


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """
    进行遥感图像分割预测
    :param file: 上传的图片
    """
    # 读取上传的图片
    image = Image.open(io.BytesIO(await file.read()))

    # 预处理（调整大小，转换成 PyTorch Tensor）
    image = image.resize((256, 256))  # 你的模型可能需要特定尺寸
    image = np.array(image).astype("float32") / 255.0
    image = torch.tensor(image).unsqueeze(0).permute(0, 3, 1, 2)  # 调整维度格式

    # 加载模型
    model = model_loader.get_model()

    # 进行预测
    with torch.no_grad():
        output = model(image)

    # 转换输出结果
    result = output.numpy().tolist()  # 转换为 JSON 兼容格式

    return {"message": "预测完成", "result": result}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

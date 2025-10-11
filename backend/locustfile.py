# locustfile.py (更新版)

import os
from locust import HttpUser, task, between
import time

# --- 配置 ---
# 使用您提供的测试用户凭据
TEST_USERNAME = "朱跃成"
TEST_PASSWORD = "21130122"
# 使用您指定的测试图片路径
TEST_IMAGE_PATH = "images/05894e69-aaea-4f41-b0b7-b55b435496e5_original.png"
# 使用您选择的 unetmamba 模型
SEGMENTATION_MODEL = "unetmamba"

class RemoteSensingUser(HttpUser):
    """
    模拟与遥感图像分割 Web 应用交互的用户。
    """
    wait_time = between(1, 3) # 任务间等待1-3秒
    host = "http://localhost:8000" # 如果后端部署在别处，请修改

    auth_token = None

    def on_start(self):
        """虚拟用户启动时执行登录"""
        self.login()

    def login(self):
        """用户登录并存储认证 token"""
        if not self.auth_token:
            try:
                response = self.client.post("/api/login", json={
                    "username": TEST_USERNAME,
                    "password": TEST_PASSWORD
                })
                response.raise_for_status()
                self.auth_token = response.json().get("access_token")
                if not self.auth_token:
                    print("登录失败：未能获取 access token。")
            except Exception as e:
                print(f"登录失败：{e}")

    def get_auth_headers(self):
        """返回包含 Authorization 的请求头"""
        if self.auth_token:
            return {"Authorization": f"Bearer {self.auth_token}"}
        else:
            print("用户未登录，无法获取认证请求头。")
            return {}

    @task(1) # 主任务
    def segment_image_workflow(self):
        """模拟核心工作流程：上传图片并执行分割"""
        if not self.auth_token:
            print("跳过分割任务：用户未登录。")
            self.login()
            return

        # 1. 上传图片
        if not os.path.exists(TEST_IMAGE_PATH):
            print(f"错误：测试图片未在路径 {TEST_IMAGE_PATH} 找到！请检查路径是否相对于 locust 命令运行的位置有效。")
            # 如果 Locust 无法直接访问这个绝对路径，您可能需要将其放在 locustfile.py 附近或提供一个脚本可以访问的路径
            return

        current_image_id = None
        try:
            with open(TEST_IMAGE_PATH, 'rb') as image_file:
                # 确定文件名，用于 multipart/form-data
                file_name = os.path.basename(TEST_IMAGE_PATH)
                # 猜测图片的 MIME 类型（或者直接指定，例如 'image/png'）
                mime_type = 'image/png' # 假设是 PNG，如果不是请修改
                upload_response = self.client.post(
                    "/upload",
                    headers=self.get_auth_headers(),
                    files={'files': (file_name, image_file, mime_type)}
                )
                upload_response.raise_for_status()
                uploaded_ids = upload_response.json().get("imageIds")
                if not uploaded_ids:
                    print("上传失败或未返回 image ID。")
                    return
                current_image_id = uploaded_ids[0]
        except Exception as e:
            print(f"图片上传失败：{e}")
            return

        # 2. 执行分割
        if not current_image_id:
            print("无法执行分割：缺少 image ID。")
            return

        try:
            segment_payload = {
                "imageIds": [current_image_id],
                "modelName": SEGMENTATION_MODEL,
                "opacities": { # 如果需要，提供默认值
                     'Background': 1.0, 'Building': 1.0, 'Road': 1.0, 'Water': 1.0,
                     'Barren': 1.0, 'Forest': 1.0, 'Agriculture': 1.0
                 }
            }
            with self.client.post(
                "/segment",
                headers=self.get_auth_headers(),
                json=segment_payload,
                name="/segment" # 在 Locust 统计中显示的名字
            ) as segment_response:
                 if segment_response.status_code != 200:
                     segment_response.failure(f"分割请求返回状态码 {segment_response.status_code}")
                 elif "results" not in segment_response.json():
                      segment_response.failure("分割响应缺少 'results' 键")

        except Exception as e:
            print(f"分割任务失败：{e}")

    # 可以根据需要添加其他 @task 方法
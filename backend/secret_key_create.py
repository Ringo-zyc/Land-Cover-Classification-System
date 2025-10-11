import secrets
import os

# 生成密钥
SECRET_KEY = secrets.token_urlsafe(32)

# 保存到 .env 文件
with open('.env', 'a') as f:
    f.write(f'SECRET_KEY={SECRET_KEY}\n')

print("密钥已生成并保存到 .env 文件:", SECRET_KEY)
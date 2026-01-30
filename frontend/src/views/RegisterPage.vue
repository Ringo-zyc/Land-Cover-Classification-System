<template>
  <div class="auth-wrapper">
    <div class="auth-container">
      <div class="auth-header">
        <div class="logo-icon">🌍</div>
        <h2>创建账户</h2>
        <p class="subtitle">加入遥感图像分割系统</p>
      </div>
      <form @submit.prevent="register" class="auth-form">
        <div class="input-group">
          <label for="username">用户名</label>
          <div class="input-wrapper">
            <span class="input-icon">👤</span>
            <input 
              type="text" 
              id="username"
              v-model="username" 
              placeholder="请设置用户名" 
              required 
            />
          </div>
        </div>
        <div class="input-group">
          <label for="password">密码</label>
          <div class="input-wrapper">
            <span class="input-icon">🔒</span>
            <input 
              type="password" 
              id="password"
              v-model="password" 
              placeholder="请设置密码（至少5位）" 
              required 
            />
          </div>
          <div class="password-strength" v-if="password.length > 0">
            <div class="strength-bar" :class="passwordStrengthClass"></div>
            <span class="strength-text">{{ passwordStrengthText }}</span>
          </div>
        </div>
        <button type="submit" class="submit-btn" :disabled="isLoading || !username || password.length < 5">
          <span v-if="!isLoading">注册</span>
          <span v-else class="loading-text">
            <span class="spinner"></span>
            注册中...
          </span>
        </button>
      </form>
      <div class="auth-footer">
        <p>已有账号？ <router-link to="/login">立即登录</router-link></p>
      </div>
    </div>
    <div class="background-decoration">
      <div class="circle circle-1"></div>
      <div class="circle circle-2"></div>
      <div class="circle circle-3"></div>
    </div>
  </div>
</template>

<script>
import axios from 'axios';
import { ElMessage } from 'element-plus';

export default {
  data() {
    return {
      username: '',
      password: '',
      isLoading: false,
    };
  },
  computed: {
    passwordStrengthClass() {
      if (this.password.length < 5) return 'weak';
      if (this.password.length < 8) return 'medium';
      return 'strong';
    },
    passwordStrengthText() {
      if (this.password.length < 5) return '密码较弱';
      if (this.password.length < 8) return '密码中等';
      return '密码强度高';
    }
  },
  methods: {
    async register() {
      if (!this.validateForm()) return;
      this.isLoading = true;
      try {
        await axios.post('http://localhost:8000/api/register', {
          username: this.username,
          password: this.password,
        });
        ElMessage.success('注册成功');
        this.$router.push('/login');
      } catch (error) {
        ElMessage.error(error.response?.data?.detail || '网络错误，请稍后重试');
      } finally {
        this.isLoading = false;
      }
    },
    validateForm() {
      if (!this.username) {
        ElMessage.error('用户名不能为空');
        return false;
      }
      if (this.password.length < 5) {
        ElMessage.error('密码长度至少为5位');
        return false;
      }
      return true;
    },
  },
};
</script>

<style scoped>
.auth-wrapper {
  min-height: 100vh;
  display: flex;
  justify-content: center;
  align-items: center;
  background: linear-gradient(135deg, #0f3460 0%, #16213e 50%, #1a1a2e 100%);
  position: relative;
  overflow: hidden;
  font-family: 'Inter', 'PingFang SC', -apple-system, BlinkMacSystemFont, sans-serif;
}

.auth-container {
  width: 100%;
  max-width: 420px;
  padding: 48px 40px;
  background: rgba(255, 255, 255, 0.95);
  backdrop-filter: blur(20px);
  border-radius: 24px;
  box-shadow: 
    0 25px 50px -12px rgba(0, 0, 0, 0.35),
    0 0 0 1px rgba(255, 255, 255, 0.1);
  position: relative;
  z-index: 10;
  animation: slideUp 0.6s ease-out;
}

@keyframes slideUp {
  from {
    opacity: 0;
    transform: translateY(30px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

.auth-header {
  text-align: center;
  margin-bottom: 36px;
}

.logo-icon {
  font-size: 48px;
  margin-bottom: 16px;
  animation: float 3s ease-in-out infinite;
}

@keyframes float {
  0%, 100% { transform: translateY(0px); }
  50% { transform: translateY(-8px); }
}

.auth-header h2 {
  font-size: 28px;
  font-weight: 700;
  color: #1a1a2e;
  margin: 0 0 8px 0;
  letter-spacing: -0.5px;
}

.subtitle {
  color: #6b7280;
  font-size: 14px;
  margin: 0;
}

.auth-form {
  display: flex;
  flex-direction: column;
  gap: 20px;
}

.input-group {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.input-group label {
  font-size: 13px;
  font-weight: 600;
  color: #374151;
  text-transform: uppercase;
  letter-spacing: 0.5px;
}

.input-wrapper {
  position: relative;
  display: flex;
  align-items: center;
}

.input-icon {
  position: absolute;
  left: 16px;
  font-size: 16px;
  opacity: 0.6;
}

.input-wrapper input {
  width: 100%;
  padding: 16px 16px 16px 48px;
  font-size: 15px;
  border: 2px solid #e5e7eb;
  border-radius: 12px;
  background: #f9fafb;
  transition: all 0.3s ease;
  color: #1f2937;
}

.input-wrapper input:focus {
  border-color: #10b981;
  background: #fff;
  outline: none;
  box-shadow: 0 0 0 4px rgba(16, 185, 129, 0.1);
}

.input-wrapper input::placeholder {
  color: #9ca3af;
}

/* Password Strength Indicator */
.password-strength {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-top: 4px;
}

.strength-bar {
  height: 4px;
  flex: 1;
  border-radius: 2px;
  transition: all 0.3s ease;
}

.strength-bar.weak {
  background: linear-gradient(90deg, #ef4444 0%, #ef4444 33%, #e5e7eb 33%);
}

.strength-bar.medium {
  background: linear-gradient(90deg, #f59e0b 0%, #f59e0b 66%, #e5e7eb 66%);
}

.strength-bar.strong {
  background: linear-gradient(90deg, #10b981 0%, #10b981 100%);
}

.strength-text {
  font-size: 12px;
  color: #6b7280;
  white-space: nowrap;
}

.submit-btn {
  margin-top: 8px;
  padding: 16px 32px;
  font-size: 16px;
  font-weight: 600;
  color: #fff;
  background: linear-gradient(135deg, #10b981 0%, #059669 100%);
  border: none;
  border-radius: 12px;
  cursor: pointer;
  transition: all 0.3s ease;
  display: flex;
  justify-content: center;
  align-items: center;
  gap: 8px;
}

.submit-btn:hover:not(:disabled) {
  transform: translateY(-2px);
  box-shadow: 0 10px 20px -5px rgba(16, 185, 129, 0.4);
}

.submit-btn:active:not(:disabled) {
  transform: translateY(0);
}

.submit-btn:disabled {
  background: linear-gradient(135deg, #9ca3af 0%, #6b7280 100%);
  cursor: not-allowed;
}

.loading-text {
  display: flex;
  align-items: center;
  gap: 8px;
}

.spinner {
  width: 18px;
  height: 18px;
  border: 2px solid transparent;
  border-top-color: #fff;
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
}

@keyframes spin {
  to { transform: rotate(360deg); }
}

.auth-footer {
  margin-top: 28px;
  text-align: center;
}

.auth-footer p {
  color: #6b7280;
  font-size: 14px;
  margin: 0;
}

.auth-footer a {
  color: #10b981;
  font-weight: 600;
  text-decoration: none;
  transition: color 0.2s;
}

.auth-footer a:hover {
  color: #059669;
  text-decoration: underline;
}

/* Background Decorations */
.background-decoration {
  position: absolute;
  inset: 0;
  z-index: 1;
  pointer-events: none;
}

.circle {
  position: absolute;
  border-radius: 50%;
  background: linear-gradient(135deg, rgba(16, 185, 129, 0.3), rgba(6, 182, 212, 0.3));
  filter: blur(60px);
}

.circle-1 {
  width: 400px;
  height: 400px;
  top: -100px;
  left: -100px;
  animation: pulse 8s ease-in-out infinite;
}

.circle-2 {
  width: 300px;
  height: 300px;
  bottom: -50px;
  right: -50px;
  animation: pulse 10s ease-in-out infinite reverse;
}

.circle-3 {
  width: 200px;
  height: 200px;
  top: 40%;
  right: 20%;
  animation: pulse 6s ease-in-out infinite;
}

@keyframes pulse {
  0%, 100% { opacity: 0.5; transform: scale(1); }
  50% { opacity: 0.8; transform: scale(1.1); }
}

/* Responsive */
@media (max-width: 480px) {
  .auth-container {
    margin: 20px;
    padding: 32px 24px;
  }
  
  .auth-header h2 {
    font-size: 24px;
  }
}
</style>
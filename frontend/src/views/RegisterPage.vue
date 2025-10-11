<template>
  <div class="auth-page">
    <h2>注册</h2>
    <form @submit.prevent="register">
      <input type="text" v-model="username" placeholder="用户名" required />
      <input type="password" v-model="password" placeholder="密码" required />
      <button type="submit" :disabled="isLoading || !username || password.length < 5">
        {{ isLoading ? '注册中...' : '注册' }}
      </button>
    </form>
    <router-link to="/login">已有账号？登录</router-link>
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
/* 与 LoginPage.vue 样式相同 */
.auth-page {
  width: clamp(350px, 80vw, 700px);
  margin: 50px auto;
  padding: 2rem;
  text-align: center;
  border: 1px solid #ccc;
  border-radius: 10px;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
  font-family: 'PingFang SC', 'Helvetica Neue', Arial, sans-serif;
  animation: fadeIn 0.5s ease-in-out;
}

@keyframes fadeIn {
  from {
    opacity: 0;
    transform: translateY(-20px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

h2 {
  margin-bottom: 1.5rem;
  font-size: 24px;
  color: #333;
}

input {
  display: block;
  width: 100%;
  margin: 1rem 0;
  padding: 0.5rem;
  font-size: 16px;
  border: 1px solid #ccc;
  border-radius: 5px;
  transition: border-color 0.3s ease;
}

input:focus {
  border-color: #007bff;
  outline: none;
  box-shadow: 0 0 5px rgba(0, 123, 255, 0.5);
}

button {
  padding: 0.5rem 2rem;
  background-color: #007bff;
  color: white;
  border: none;
  border-radius: 5px;
  cursor: pointer;
  font-size: 16px;
  transition: background-color 0.3s ease, transform 0.2s ease;
}

button:hover:not(:disabled) {
  background-color: #0056b3;
  transform: scale(1.05);
}

button:disabled {
  background-color: #ccc;
  cursor: not-allowed;
}

router-link {
  color: #007bff;
  text-decoration: none;
}
</style>
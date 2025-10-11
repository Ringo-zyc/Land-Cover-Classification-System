<template>
  <div class="ai-assistant-panel" v-if="showPanel">
    <div class="ai-panel-header">
      <h3>AI 助手</h3>
      <button class="btn close-panel-btn" @click="togglePanel">
        <i class="iconfont icon-close"></i>
      </button>
    </div>
    <div class="ai-panel-content">
      <button class="btn ai-feature-btn" @click="toggleReportInput">
        <i class="iconfont icon-report"></i> 生成图像分析报告
      </button>
      <transition name="slide-fade">
        <div v-if="showReportInput" class="prompt-input-container">
          <label for="report-prompt">AI提示词：</label>
          <textarea
            id="report-prompt"
            v-model="reportPrompt"
            rows="6"
            placeholder="请输入报告生成提示词"
          ></textarea>
          <button class="btn generate-btn generate-report-btn" @click="generateReport">生成报告</button>
        </div>
      </transition>

      <button class="btn ai-feature-btn" @click="toggleStyleInput">
        <i class="iconfont icon-style"></i> 风格化图像创作
      </button>
      <transition name="slide-fade">
        <div v-if="showStyleInput" class="style-input-container">
          <label for="style-prompt">风格提示词：</label>
          <input
            id="style-prompt"
            v-model="stylePrompt"
            placeholder="例如：油画风格"
          />
          <button class="btn generate-btn generate-style-btn" @click="generateStyledImage">生成图像</button>
        </div>
      </transition>

      <button class="btn ai-feature-btn" @click="applyStyleTransfer">
        <i class="iconfont icon-magic"></i> 风格迁移
      </button>

      <div v-if="isLoading" class="ai-loading">
        <div class="loader"></div> 正在处理，请稍候...
      </div>

      <div v-if="report" class="ai-report-display">
        <h4>分析报告</h4>
        <pre>{{ report }}</pre>
        <button class="btn export-btn" @click="exportReport">导出报告</button>
      </div>

      <div v-if="styledImageUrl" class="image-display">
        <h4>风格化图像</h4>
        <img :src="styledImageUrl" alt="风格化图像" />
        <button class="btn download-btn" @click="downloadImage">下载图像</button>
      </div>
    </div>
  </div>
</template>

<script>
import axios from "axios";
import { ElMessage } from "element-plus";
import { saveAs } from "file-saver";

export default {
  name: "AiAssistantPanel",
  props: {
    showPanel: Boolean,
    segmentationStats: Array,
    currentIndex: Number,
    uploadedImages: Array,
  },
  data() {
    return {
      showReportInput: false,
      showStyleInput: false,
      reportPrompt: "",
      stylePrompt: "",
      report: "",
      styledImageUrl: "",
      isLoading: false,
    };
  },
  methods: {
    togglePanel() {
      this.$emit("update:showPanel", !this.showPanel);
    },
    toggleReportInput() {
      this.showReportInput = !this.showReportInput;
      if (this.showReportInput) this.showStyleInput = false;
    },
    toggleStyleInput() {
      this.showStyleInput = !this.showStyleInput;
      if (this.showStyleInput) this.showReportInput = false;
    },
    async generateReport() {
      if (!this.reportPrompt) {
        ElMessage.warning("请输入报告提示词");
        return;
      }
      this.isLoading = true;
      if (!this.segmentationStats || !this.segmentationStats[this.currentIndex]) {
        ElMessage.warning("请先完成图像分割以获取统计数据");
        this.isLoading = false;
        return;
      }
      const currentStats = this.segmentationStats[this.currentIndex];
      try {
        const response = await axios.post(
          "http://localhost:8000/api/generate_report",
          { prompt: this.reportPrompt, segmentationStats: currentStats },
          { headers: { Authorization: `Bearer ${localStorage.getItem("token")}` } }
        );
        this.report = response.data.report;
        ElMessage.success("报告生成成功");
      } catch (error) {
        ElMessage.error("报告生成失败");
        console.error("生成报告错误:", error);
      } finally {
        this.isLoading = false;
      }
    },
    async generateStyledImage() {
      if (!this.stylePrompt) {
        ElMessage.warning("请输入风格提示词");
        return;
      }
      this.isLoading = true;
      if (!this.uploadedImages || this.uploadedImages.length === 0 || this.currentIndex >= this.uploadedImages.length) {
        ElMessage.warning("请先上传并选择一张图像");
        this.isLoading = false;
        return;
      }
      const currentFile = this.uploadedImages[this.currentIndex];
      if (!currentFile) {
        ElMessage.error("未找到当前选中的图像文件");
        this.isLoading = false;
        return;
      }
      const formData = new FormData();
      formData.append("file", currentFile);
      formData.append("stylePrompt", this.stylePrompt);
      formData.append("segmentationStats", JSON.stringify(this.segmentationStats[this.currentIndex] || {}));
      try {
        const response = await axios.post(
          "http://localhost:8000/api/generate_styled_image",
          formData,
          {
            headers: {
              Authorization: `Bearer ${localStorage.getItem("token")}`,
              "Content-Type": "multipart/form-data",
            },
          }
        );
        const baseUrl = "http://localhost:8000";
        const imagePath = response.data.styledImageUrl;
        this.styledImageUrl = imagePath.startsWith("http") ? imagePath : baseUrl + imagePath;
        ElMessage.success("风格化图像生成成功");
      } catch (error) {
        ElMessage.error("风格化图像生成失败");
        console.error("风格化图像生成错误:", error);
      } finally {
        this.isLoading = false;
      }
    },
    async applyStyleTransfer() {
      this.isLoading = true;
      if (!this.uploadedImages || this.uploadedImages.length === 0 || this.currentIndex >= this.uploadedImages.length) {
        ElMessage.warning("请先上传并选择一张图像");
        this.isLoading = false;
        return;
      }
      const currentFile = this.uploadedImages[this.currentIndex];
      if (!currentFile) {
        ElMessage.error("未找到当前选中的图像文件");
        this.isLoading = false;
        return;
      }
      const formData = new FormData();
      formData.append("file", currentFile);
      try {
        const response = await axios.post(
          "http://localhost:8000/api/style_transfer?style_type=default",
          formData,
          {
            headers: {
              Authorization: `Bearer ${localStorage.getItem("token")}`,
              "Content-Type": "multipart/form-data",
            },
          }
        );
        const baseUrl = "http://localhost:8000";
        const imagePath = response.data.styledImageUrl;
        this.styledImageUrl = imagePath.startsWith("http") ? imagePath : baseUrl + imagePath;
        ElMessage.success("风格迁移成功");
      } catch (error) {
        ElMessage.error("风格迁移失败");
        console.error("风格迁移错误:", error);
      } finally {
        this.isLoading = false;
      }
    },
    exportReport() {
      if (this.report) {
        const blob = new Blob([this.report], { type: "text/plain;charset=utf-8" });
        saveAs(blob, "analysis_report.txt");
      } else {
        ElMessage.warning("请先生成报告");
      }
    },
    downloadImage() {
      if (this.styledImageUrl) {
        saveAs(this.styledImageUrl, "styled_image.png");
      } else {
        ElMessage.warning("请先生成风格化图像或风格迁移图像");
      }
    },
  },
};
</script>

<style scoped>
/* 莫兰迪色系 AI 助手面板样式 */
.ai-assistant-panel {
  position: fixed;
  top: 0;
  right: 0;
  width: 350px;
  height: 100%;
  background-color: #f5f5f7; /* Homepage background color */
  box-shadow: -2px 0 5px rgba(0, 0, 0, 0.1);
  z-index: 101;
  transition: transform 0.3s ease-out;
  font-family: 'WenQuanYi Micro Hei', sans-serif; /* 统一字体 */
}

.ai-panel-header {
  padding: 15px;
  border-bottom: 1px solid #e0e0e0;
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.ai-panel-header h3 {
  margin: 0;
  font-size: 1.1em; /* 调整标题字号 */
  color: #333; /* 标题文字颜色 */
}

.close-panel-btn {
  background-color: transparent;
  color: #777;
  padding: 8px;
  border-radius: 50%;
  border: none;
  transition: background-color 0.2s ease;
}

.close-panel-btn:hover {
  background-color: #e0e0e0; /* Hover时更浅的灰色 */
  color: #555;
}

.ai-panel-content {
  padding: 25px;
  display: flex;
  flex-direction: column;
  gap: 15px; /* 调整内容间距 */
}

.ai-feature-btn {
  background-color: #e8e8ed; /* 按钮背景色，浅灰色 */
  color: #555; /* 按钮文字颜色，深灰色 */
  padding: 10px 15px; /* 调整按钮padding */
  border-radius: 8px;
  text-align: left;
  display: flex;
  align-items: center;
  gap: 8px;
  border: none;
  transition: background-color 0.2s ease, color 0.2s ease;
}

.ai-feature-btn:hover {
  background-color: #d2d2d7; /* Hover时更浅的灰色 */
  color: #333; /* Hover时文字颜色加深 */
}

.prompt-input-container,
.style-input-container {
  display: flex;
  flex-direction: column;
  gap: 8px; /* 调整输入框容器间距 */
}

.prompt-input-container label,
.style-input-container label {
  font-weight: bold;
  color: #555; /* 标签文字颜色 */
  font-size: 0.95em; /* 调整标签字号 */
}

.prompt-input-container textarea,
.style-input-container input {
  border: 1px solid #d2d2d7;
  border-radius: 8px;
  padding: 10px;
  font-size: 14px;
  box-shadow: none; /* 移除input阴影 */
  transition: border-color 0.2s ease;
}

.prompt-input-container textarea:focus,
.style-input-container input:focus {
  border-color: #a0a0a5; /* Focus时边框颜色 */
  outline: none;
  box-shadow: none; /* 移除focus阴影 */
}

.generate-btn {
  background-color: rgb(155, 158, 137); /* AI助手按钮颜色，homepage中的莫兰迪绿灰 */
  color: #fff;
  padding: 10px 20px;
  border-radius: 8px;
  border: none;
  cursor: pointer;
  transition: background-color 0.3s ease, transform 0.2s ease;
  font-family: 'WenQuanYi Micro Hei', sans-serif;
  font-weight: bold;
  font-size: 0.9em; /* 调整按钮字号 */
}

.generate-btn:hover {
  background-color: rgb(135, 138, 117); /* Hover时颜色略微加深 */
  transform: scale(1.02);
  box-shadow: 0 2px 6px rgba(0, 0, 0, 0.1); /* Hover时阴影 */
}


.ai-loading {
  display: flex;
  align-items: center;
  gap: 10px;
  color: #777;
  font-size: 0.9em; /* 调整加载提示字号 */
}

.loader {
  border: 3px solid #f3f3f3;
  border-top: 3px solid rgb(155, 158, 137); /* 加载动画颜色，与按钮颜色一致 */
  border-radius: 50%;
  width: 16px;
  height: 16px;
  animation: spin 1s linear infinite;
}

@keyframes spin {
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
}

.ai-report-display,
.image-display {
  padding: 12px; /* 调整显示区域padding */
  border: 1px solid #e0e0e0;
  border-radius: 8px;
  background-color: #f9f9f9;
  box-shadow: none; /* 移除显示区域阴影 */
  max-height: 250px; /* 调整最大高度 */
  overflow-y: auto;
  font-size: 0.9em; /* 调整内容字号 */
}

.ai-report-display h4,
.image-display h4 {
  margin: 0 0 8px 0; /* 调整标题margin */
  font-size: 1em;
  color: #555; /* 显示区域标题颜色 */
}

.ai-report-display pre {
  white-space: pre-wrap;
  margin: 0;
  font-family: monospace;
  color: #333;
  font-size: 0.9em; /* 调整 pre 代码字号 */
}

.image-display img {
  max-width: 100%;
  height: auto;
  border-radius: 4px; /* 图片圆角 */
}

.export-btn,
.download-btn {
  background-color: rgb(153, 164, 188); /* 导出/下载按钮颜色，homepage中的莫兰迪蓝灰 */
  color: #fff;
  padding: 8px 16px;
  border-radius: 8px;
  border: none;
  cursor: pointer;
  transition: background-color 0.3s ease, transform 0.2s ease;
  font-family: 'WenQuanYi Micro Hei', sans-serif;
  font-weight: bold;
  font-size: 0.85em; /* 调整按钮字号 */
}

.export-btn:hover,
.download-btn:hover {
  background-color: rgb(133, 144, 168); /* Hover时颜色略微加深 */
  opacity: 1;
  transform: scale(1.02);
  box-shadow: 0 2px 6px rgba(0, 0, 0, 0.1); /* Hover时阴影 */
}

.slide-fade-enter-active,
.slide-fade-leave-active {
  transition: all 0.3s ease;
}
.slide-fade-enter-from,
.slide-fade-leave-to {
  transform: translateY(-10px);
  opacity: 0;
}
</style>
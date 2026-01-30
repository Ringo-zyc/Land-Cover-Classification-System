<template>
  <div class="container">
    <h1>遥感图像分割系统</h1>
    <div class="header-controls">
      <div class="model-select">
        <label for="model">选择模型：</label>
        <select v-model="selectedModel" id="model">
          <option value="unetmamba">UNetMamba</option>
          <option value="dcswin_small">DCSwin Small</option>
          <option value="unetformer_r18">UNetFormer R18</option>
        </select>
      </div>
      <button class="btn batch-btn" @click="toggleBatchMode">
        {{ isBatchMode ? "返回单张" : "批量处理" }}
      </button>
    </div>
    <div class="image-display" :class="{ 'batch-mode': isBatchMode }">
      <div v-if="!isBatchMode" class="single-image">
        <div class="image-pair">
          <div
            class="image-container"
            @click="selectImage(0)"
            @dragover.prevent
            @drop="handleDrop(0, $event)"
          >
            <img :src="images[0].originalImage" alt="原始图像" v-if="images[0].originalImage" />
            <span v-else>点击或拖动上传图像</span>
          </div>
          <div class="image-container">
            <img
              :src="images[0].segmentedImage"
              alt="分割图像"
              v-if="images[0].segmentedImage"
              @click="exportSegmentedImage(0)"
            />
            <span v-else>分割后图像</span>
          </div>
        </div>
      </div>
      <div v-else class="batch-images">
        <div class="image-section">
          <h3>上传图像</h3>
          <div class="image-row">
            <div
              v-for="(image, index) in images.slice(0, 4)"
              :key="index"
              class="image-container"
              @click="selectImage(index)"
              @dragover.prevent
              @drop="handleDrop(index, $event)"
              :class="{ 'selected': currentIndex === index }"
            >
              <img :src="image.originalImage" alt="原始图像" v-if="image.originalImage" />
              <span v-else>点击或拖动上传图像 {{ index + 1 }}</span>
            </div>
          </div>
        </div>
        <div class="image-section">
          <h3>分割结果</h3>
          <div class="image-row">
            <div
              v-for="(image, index) in images.slice(0, 4)"
              :key="index"
              class="image-container"
            >
              <img
                :src="image.segmentedImage"
                alt="分割图像"
                v-if="image.segmentedImage"
                @click="exportSegmentedImage(index)"
              />
              <span v-else>分割后图像 {{ index + 1 }}</span>
              <button class="btn zoom-btn" @click="openModal(index)">放大</button>
            </div>
          </div>
        </div>
      </div>
    </div>
    <div v-if="showModal" class="modal" @click="closeModal">
      <div class="modal-content" @click.stop>
        <div class="modal-images">
          <div class="modal-image-container">
            <img :src="modalImage.originalImage" alt="原始图像" v-if="modalImage.originalImage" />
            <span v-else>无图像</span>
          </div>
          <div class="modal-image-container">
            <img :src="modalImage.segmentedImage" alt="分割图像" v-if="modalImage.segmentedImage" />
            <span v-else>无图像</span>
          </div>
        </div>
        <div class="modal-stats" v-if="segmentationStats[currentIndex]">
          <div class="chart-item">
            <PieChart
              :chartData="segmentationStats[currentIndex]"
              :classColors="classColors"
              ref="modalPieChart"
            />
            <button class="btn export-btn" @click="exportChart('pie', currentIndex)">导出饼图</button>
          </div>
          <div class="chart-item">
            <HistogramChart
              :chartData="segmentationStats[currentIndex]"
              :classColors="classColors"
              ref="modalHistogramChart"
            />
            <button class="btn export-btn" @click="exportChart('histogram', currentIndex)">导出条形图</button>
          </div>
        </div>
        <button class="btn close-btn" @click="closeModal">关闭</button>
      </div>
    </div>
    <div class="stats" v-if="!isBatchMode && segmentationStats.length > 0">
      <h3>分割数据统计</h3>
      <div class="stats-container">
        <div class="chart-item">
          <PieChart
            v-if="segmentationStats[0]"
            ref="pieChart"
            :chartData="segmentationStats[0]"
            :classColors="classColors"
          />
          <button class="btn export-btn" @click="exportChart('pie')">导出饼图</button>
        </div>
        <div class="chart-item">
          <HistogramChart
            v-if="segmentationStats[0]"
            ref="histogramChart"
            :chartData="segmentationStats[0]"
            :classColors="classColors"
          />
          <button class="btn export-btn" @click="exportChart('histogram')">导出条形图</button>
        </div>
      </div>
    </div>
    <div class="color-legend">
      <h3>颜色标注</h3>
      <ul>
        <li v-for="(color, key) in classColors" :key="key">
          <span class="color-box" :style="{ backgroundColor: color }"></span> {{ key }}
        </li>
      </ul>
    </div>
    <input
      type="file"
      id="image-upload"
      accept="image/*"
      @change="handleFileSelect"
      style="display: none"
      :multiple="isBatchMode"
    />
    <div class="loading" v-if="isProcessing">正在处理，请稍候...</div>
    <HistoryPanel :classColors="classColors" />
    <div class="horizontal-action-bar">
      <button class="action-button ai-assistant-button" @click="toggleAiAssistantPanel">
        <i class="iconfont icon-ai"></i>
        <span>AI助手</span>
      </button>
      <button class="action-button overlay-button" @click="openOverlayModal">
        <i class="iconfont icon-overlay"></i>
        <span>叠加</span>
      </button>
      <button class="action-button clear-button" @click="clearAll">
        <i class="iconfont icon-clear"></i>
        <span>清空</span>
      </button>
    </div>
    <transition name="fade">
      <div class="modal-backdrop" v-if="showAiAssistantPanel" @click.self="showAiAssistantPanel = false"></div>
    </transition>
    <transition name="ai-modal">
      <AiAssistantPanel
        v-if="showAiAssistantPanel"
        class="ai-assistant-modal"
        v-model:showPanel="showAiAssistantPanel"
        :segmentationStats="segmentationStats"
        :currentIndex="currentIndex"
        :uploadedImages="selectedFiles"
      />
    </transition>
    <div v-if="showOverlayModal" class="modal" @click="closeOverlayModal">
      <div class="modal-content overlay-modal-content" @click.stop>
        <h3>叠加图像</h3>
        <div v-if="isBatchMode" class="image-selection">
          <div
            v-for="(image, index) in images.slice(0, 4)"
            :key="index"
            class="thumbnail-container"
            :class="{ selected: selectedOverlayIndex === index }"
            @click="selectOverlayImage(index)"
          >
            <img :src="image.originalImage" alt="缩略图" class="thumbnail" />
            <span>图片 {{ index + 1 }}</span>
          </div>
        </div>
        <div class="overlay-container">
          <img :src="selectedOverlayImage.originalImage" alt="原图" class="overlay-image" />
          <img
            :src="selectedOverlayImage.segmentedImage"
            alt="分割图"
            class="overlay-image"
            :style="{ opacity: overlayOpacity }"
          />
        </div>
        <div class="opacity-slider">
          <label>调整透明度</label>
          <input type="range" min="0" max="1" step="0.01" v-model="overlayOpacity" />
          <span>{{ (overlayOpacity * 100).toFixed(0) }}%</span>
        </div>
        <button class="btn close-btn" @click="closeOverlayModal">关闭</button>
      </div>
    </div>
  </div>
</template>

<script>
import axios from "axios";
import { ElMessage } from "element-plus";
import PieChart from "../components/PieChart.vue";
import HistogramChart from "../components/HistogramChart.vue";
import HistoryPanel from "../components/HistoryPanel.vue";
import AiAssistantPanel from "../components/AiAssistantPanel.vue";
import { saveAs } from "file-saver";

export default {
  components: { PieChart, HistogramChart, HistoryPanel, AiAssistantPanel },
  data() {
    return {
      selectedModel: "unetmamba",
      isBatchMode: false,
      images: [{ originalImage: "", segmentedImage: "", uploadedImageId: null }],
      segmentationStats: [],
      isProcessing: false,
      currentIndex: 0,
      selectedFiles: [],
      showModal: false,
      modalImage: { originalImage: "", segmentedImage: "" },
      classColors: {
        Background: "rgb(255, 255, 255)",
        Building: "rgb(255, 0, 0)",
        Road: "rgb(255, 255, 0)",
        Water: "rgb(0, 0, 255)",
        Barren: "rgb(159, 129, 183)",
        Forest: "rgb(0, 255, 0)",
        Agriculture: "rgb(255, 195, 128)",
      },
      showOverlayModal: false,
      overlayOpacity: 0.5,
      selectedOverlayIndex: 0,
      showAiAssistantPanel: false,
      styledImageUrl: "", // Added to store the stylized image URL
    };
  },
  computed: {
    selectedOverlayImage() {
      return this.isBatchMode ? this.images[this.selectedOverlayIndex] : this.images[0];
    },
  },
  methods: {
    toggleBatchMode() {
      this.isBatchMode = !this.isBatchMode;
      if (!this.isBatchMode) {
        this.currentIndex = 0;
      }
      this.clearAll();
    },
    selectImage(index) {
      this.currentIndex = index;
      document.getElementById("image-upload").click();
    },
    handleFileSelect(event) {
      const files = event.target.files;
      if (!files.length) return;
      if (this.isBatchMode && files.length > 4) {
        ElMessage.error("批量模式最多上传4张图像");
        return;
      }
      if (!this.isBatchMode && files.length > 1) {
        ElMessage.error("单张模式只能上传1张图像");
        return;
      }
      this.selectedFiles = Array.from(files);
      if (this.isBatchMode) {
        this.images = this.selectedFiles.map((file) => ({
          originalImage: URL.createObjectURL(file),
          segmentedImage: "",
          uploadedImageId: null,
        }));
      } else {
        this.images = [
          {
            originalImage: URL.createObjectURL(files[0]),
            segmentedImage: "",
            uploadedImageId: null,
          },
        ];
      }
      this.uploadAndSegment();
    },
    handleDrop(index, event) {
      event.preventDefault();
      const file = event.dataTransfer.files[0];
      if (file) {
        this.selectedFiles = [file];
        if (!this.isBatchMode) {
          index = 0;
        }
        this.images[index] = {
          originalImage: URL.createObjectURL(file),
          segmentedImage: "",
          uploadedImageId: null,
        };
        this.uploadAndSegment();
      }
    },
    async uploadAndSegment() {
      const token = localStorage.getItem("token");
      if (!token) {
        ElMessage.error("请先登录");
        this.$router.push("/login");
        return;
      }

      if (!this.selectedFiles.length) {
        ElMessage.error("未选择任何文件");
        return;
      }

      this.isProcessing = true;
      const formData = new FormData();
      this.selectedFiles.forEach((file) => formData.append("files", file));

      try {
        const uploadResponse = await axios.post("http://localhost:8000/upload", formData, {
          headers: {
            "Content-Type": "multipart/form-data",
            Authorization: `Bearer ${token}`,
          },
        });
        const imageIds = uploadResponse.data.imageIds;
        this.images.forEach((image, index) => {
          if (index < imageIds.length) image.uploadedImageId = imageIds[index];
        });

        const segmentResponse = await axios.post(
          "http://localhost:8000/segment",
          { imageIds, modelName: this.selectedModel },
          {
            headers: {
              Authorization: `Bearer ${token}`,
            },
          }
        );

        this.images.forEach((image, index) => {
          if (image.uploadedImageId && segmentResponse.data.results[index]) {
            image.segmentedImage = `http://localhost:8000${segmentResponse.data.results[index].imageUrl}`;
            this.segmentationStats[index] = segmentResponse.data.results[index].stats;
          }
        });

        if (!this.isBatchMode) {
          this.segmentationStats = [this.segmentationStats[0]];
        }
        ElMessage.success("分割完成");
      } catch (error) {
        console.error("上传或分割失败:", error);
        if (error.response && error.response.status === 401) {
          ElMessage.error("认证失败，请重新登录");
          localStorage.removeItem("token");
          this.$router.push("/login");
        } else {
          ElMessage.error(`操作失败: ${error.response?.data?.detail || "未知错误"}`);
        }
      } finally {
        this.isProcessing = false;
      }
    },
    clearAll() {
      this.images = this.isBatchMode
        ? Array(4).fill().map(() => ({ originalImage: "", segmentedImage: "", uploadedImageId: null }))
        : [{ originalImage: "", segmentedImage: "", uploadedImageId: null }];
      this.segmentationStats = [];
      this.selectedFiles = [];
      this.styledImageUrl = ""; // Reset stylized image URL
    },
    openModal(index) {
      this.currentIndex = index;
      this.modalImage = {
        originalImage: this.images[index].originalImage,
        segmentedImage: this.images[index].segmentedImage,
      };
      this.showModal = true;
    },
    closeModal() {
      this.showModal = false;
    },
    exportChart(type, index) {
      let chartCanvas;
      if (type === "pie") {
        chartCanvas =
          this.$refs.pieChart?.$refs.chartCanvas || this.$refs.modalPieChart?.$refs.chartCanvas;
      } else if (type === "histogram") {
        chartCanvas =
          this.$refs.histogramChart?.$refs.chartCanvas ||
          this.$refs.modalHistogramChart?.$refs.chartCanvas;
      }
      if (chartCanvas) {
        const link = document.createElement("a");
        link.href = chartCanvas.toDataURL("image/png");
        link.download = `${type}_chart_image_${index || 0}.png`;
        link.click();
      } else {
        ElMessage.error("图表尚未渲染，无法导出");
      }
    },
    exportSegmentedImage(index) {
      const image = this.images[index];
      if (image.segmentedImage) {
        saveAs(image.segmentedImage, `segmented_image_${index}.png`);
      } else {
        ElMessage.warning("没有可导出的分割图像");
      }
    },
    openOverlayModal() {
      if (this.isBatchMode) {
        if (!this.images.some((img) => img.originalImage && img.segmentedImage)) {
          ElMessage.warning("请先上传并分割图像");
          return;
        }
      } else {
        if (!this.images[0].originalImage || !this.images[0].segmentedImage) {
          ElMessage.warning("请先上传并分割图像");
          return;
        }
      }
      this.showOverlayModal = true;
    },
    closeOverlayModal() {
      this.showOverlayModal = false;
    },
    selectOverlayImage(index) {
      this.selectedOverlayIndex = index;
    },
    toggleAiAssistantPanel() {
      this.showAiAssistantPanel = !this.showAiAssistantPanel;
    },
    async applyStyleTransfer() {
      const token = localStorage.getItem("token");
      if (!token) {
        ElMessage.error("请先登录");
        this.$router.push("/login");
        return;
      }

      if (!this.images[this.currentIndex].originalImage) {
        ElMessage.error("请先上传图像");
        return;
      }

      this.isProcessing = true;
      const formData = new FormData();
      const file = this.selectedFiles[this.currentIndex];
      formData.append("file", file);

      try {
        const response = await axios.post(
          "http://localhost:8000/api/style_transfer?style_type=default",
          formData,
          {
            headers: {
              Authorization: `Bearer ${token}`,
              "Content-Type": "multipart/form-data",
            },
          }
        );
        const baseUrl = "http://localhost:8000";
        const imagePath = response.data.styledImageUrl;
        if (imagePath) {
          this.styledImageUrl = baseUrl + imagePath;
          ElMessage.success("风格迁移完成");
        } else {
          ElMessage.error("后端未返回风格迁移图像路径");
        }
      } catch (error) {
        console.error("风格迁移失败:", error);
        ElMessage.error(`风格迁移失败: ${error.response?.data?.detail || "未知错误"}`);
      } finally {
        this.isProcessing = false;
      }
    },
  },
  mounted() {
    axios.interceptors.response.use(
      (response) => response,
      (error) => {
        if (error.response && error.response.status === 401) {
          ElMessage.error("认证失败，请重新登录");
          localStorage.removeItem("token");
          this.$router.push("/login");
        }
        return Promise.reject(error);
      }
    );

    if (!localStorage.getItem("token")) {
      ElMessage.warning("请先登录以使用完整功能");
    }
  },
};
</script>

<style scoped>
/* Modern Design System */
:root {
  --primary: #3b82f6;
  --primary-dark: #1d4ed8;
  --success: #10b981;
  --warning: #f59e0b;
  --danger: #ef4444;
  --bg-dark: #0f172a;
  --bg-card: rgba(255, 255, 255, 0.95);
  --text-primary: #1e293b;
  --text-secondary: #64748b;
  --border: #e2e8f0;
  --shadow: 0 10px 40px -10px rgba(0, 0, 0, 0.15);
}

.container {
  max-width: 1600px;
  margin: 0 auto;
  padding: 32px;
  background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
  border-radius: 0;
  min-height: 100vh;
  position: relative;
  font-family: 'Inter', 'PingFang SC', -apple-system, BlinkMacSystemFont, sans-serif;
}

h1 {
  text-align: center;
  font-size: 32px;
  font-weight: 700;
  color: #fff;
  margin-bottom: 8px;
  letter-spacing: -0.5px;
  background: linear-gradient(135deg, #60a5fa, #a78bfa);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}

.header-controls {
  display: flex;
  justify-content: center;
  align-items: center;
  gap: 24px;
  margin-bottom: 32px;
  padding: 20px 24px;
  background: rgba(255, 255, 255, 0.08);
  backdrop-filter: blur(20px);
  border-radius: 16px;
  border: 1px solid rgba(255, 255, 255, 0.1);
}

.model-select {
  display: flex;
  align-items: center;
  gap: 12px;
}

.model-select label {
  color: #94a3b8;
  font-size: 14px;
  font-weight: 500;
}

.model-select select {
  padding: 12px 20px;
  border-radius: 10px;
  border: 2px solid rgba(255, 255, 255, 0.1);
  background: rgba(255, 255, 255, 0.1);
  color: #fff;
  font-size: 14px;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.3s ease;
  backdrop-filter: blur(10px);
}

.model-select select:hover,
.model-select select:focus {
  border-color: var(--primary);
  background: rgba(59, 130, 246, 0.2);
  outline: none;
}

.model-select select option {
  background: #1e293b;
  color: #fff;
}

.btn {
  padding: 12px 28px;
  border-radius: 10px;
  border: none;
  cursor: pointer;
  transition: all 0.3s ease;
  font-family: inherit;
  font-weight: 600;
  font-size: 14px;
  color: #fff;
  position: relative;
  overflow: hidden;
}

.btn::before {
  content: '';
  position: absolute;
  inset: 0;
  background: linear-gradient(135deg, rgba(255,255,255,0.1), transparent);
  opacity: 0;
  transition: opacity 0.3s;
}

.btn:hover::before {
  opacity: 1;
}

.btn:hover {
  transform: translateY(-2px);
  box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
}

.btn:active {
  transform: translateY(0);
}

.batch-btn {
  background: linear-gradient(135deg, #6366f1, #8b5cf6);
}

.image-display {
  display: flex;
  flex-direction: column;
  gap: 32px;
  align-items: center;
}

.single-image .image-pair {
  display: flex;
  justify-content: center;
  gap: 32px;
}

.image-container {
  background: linear-gradient(180deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0.05) 100%);
  border-radius: 20px;
  overflow: hidden;
  box-shadow: 
    0 20px 40px -10px rgba(0, 0, 0, 0.3),
    inset 0 1px 1px rgba(255, 255, 255, 0.1);
  display: flex;
  align-items: center;
  justify-content: center;
  position: relative;
  transition: all 0.3s ease;
  border: 1px solid rgba(255, 255, 255, 0.1);
  color: #64748b;
  font-size: 14px;
  cursor: pointer;
}

.image-container:hover {
  transform: translateY(-4px);
  box-shadow: 
    0 30px 50px -15px rgba(0, 0, 0, 0.4),
    inset 0 1px 1px rgba(255, 255, 255, 0.2);
  border-color: rgba(59, 130, 246, 0.3);
}

.single-image .image-container {
  width: 550px;
  height: 550px;
}

.batch-images .image-container {
  width: 320px;
  height: 320px;
}

.image-container img {
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.image-row {
  display: flex;
  gap: 24px;
  justify-content: center;
  flex-wrap: wrap;
}

.image-section h3 {
  color: #e2e8f0;
  font-size: 18px;
  font-weight: 600;
  margin-bottom: 16px;
  text-align: center;
}

.zoom-btn {
  position: absolute;
  bottom: 12px;
  right: 12px;
  padding: 8px 16px;
  font-size: 13px;
  background: linear-gradient(135deg, var(--primary), var(--primary-dark));
  border-radius: 8px;
}

.modal {
  position: fixed;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background-color: rgba(0, 0, 0, 0.75);
  backdrop-filter: blur(8px);
  display: flex;
  justify-content: center;
  align-items: center;
  z-index: 1000;
}

.modal-content {
  background: linear-gradient(180deg, #1e293b, #0f172a);
  padding: 28px;
  border-radius: 20px;
  display: flex;
  flex-direction: column;
  gap: 24px;
  width: 90%;
  max-width: 1000px;
  border: 1px solid rgba(255, 255, 255, 0.1);
  box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
  animation: modalSlideUp 0.4s ease-out;
}

@keyframes modalSlideUp {
  from {
    opacity: 0;
    transform: translateY(30px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

.modal-images {
  display: flex;
  gap: 20px;
  justify-content: center;
}

.modal-image-container {
  width: 400px;
  height: 400px;
  border-radius: 16px;
  overflow: hidden;
  background: rgba(255, 255, 255, 0.05);
  display: flex;
  align-items: center;
  justify-content: center;
  border: 1px solid rgba(255, 255, 255, 0.1);
}

.modal-image-container img {
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.modal-stats {
  display: flex;
  gap: 24px;
  justify-content: center;
}

/* Color Legend */
.color-legend {
  position: fixed;
  top: 100px;
  right: 24px;
  background: rgba(15, 23, 42, 0.95);
  backdrop-filter: blur(20px);
  padding: 20px;
  border-radius: 16px;
  border: 1px solid rgba(255, 255, 255, 0.1);
  box-shadow: 0 10px 40px -10px rgba(0, 0, 0, 0.3);
  z-index: 50;
}

.color-legend h3 {
  color: #e2e8f0;
  font-size: 14px;
  font-weight: 600;
  margin: 0 0 16px 0;
  text-transform: uppercase;
  letter-spacing: 0.5px;
}

.color-legend ul {
  list-style: none;
  padding: 0;
  margin: 0;
}

.color-legend li {
  display: flex;
  align-items: center;
  margin-bottom: 10px;
  color: #94a3b8;
  font-size: 13px;
}

.color-box {
  width: 18px;
  height: 18px;
  margin-right: 10px;
  border-radius: 5px;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
}

.loading {
  text-align: center;
  color: #60a5fa;
  margin: 24px 0;
  font-size: 15px;
  animation: pulse 1.5s ease-in-out infinite;
}

@keyframes pulse {
  0%, 100% { opacity: 0.6; }
  50% { opacity: 1; }
}

.stats {
  background: rgba(255, 255, 255, 0.05);
  border-radius: 20px;
  padding: 28px;
  margin-top: 32px;
  border: 1px solid rgba(255, 255, 255, 0.1);
}

.stats h3 {
  color: #e2e8f0;
  font-size: 18px;
  font-weight: 600;
  text-align: center;
  margin: 0 0 20px 0;
}

.stats-container {
  display: flex;
  justify-content: center;
  align-items: flex-start;
  gap: 32px;
}

.chart-item {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 12px;
}

.export-btn {
  background: linear-gradient(135deg, var(--success), #059669);
  color: #fff;
  padding: 10px 20px;
  border-radius: 8px;
  border: none;
  cursor: pointer;
  font-weight: 600;
  font-size: 13px;
  transition: all 0.3s;
}

.export-btn:hover {
  transform: translateY(-2px);
  box-shadow: 0 8px 20px -5px rgba(16, 185, 129, 0.4);
}

/* Floating Action Buttons */
.horizontal-action-bar {
  position: fixed;
  bottom: 32px;
  right: 32px;
  display: flex;
  gap: 16px;
  z-index: 100;
}

.action-button {
  width: 72px;
  height: 72px;
  border-radius: 50%;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 0;
  color: #fff;
  border: none;
  cursor: pointer;
  box-shadow: 0 8px 25px -5px rgba(0, 0, 0, 0.3);
  transition: all 0.3s ease;
  font-family: inherit;
  font-weight: 600;
}

.action-button:hover {
  transform: translateY(-4px) scale(1.05);
  box-shadow: 0 15px 35px -8px rgba(0, 0, 0, 0.4);
}

.action-button .iconfont {
  font-size: 24px;
  margin-bottom: 4px;
}

.action-button span {
  font-size: 11px;
  font-weight: 500;
}

.ai-assistant-button {
  background: linear-gradient(135deg, #8b5cf6, #6366f1);
}

.overlay-button {
  background: linear-gradient(135deg, #3b82f6, #0ea5e9);
}

.clear-button {
  background: linear-gradient(135deg, #f43f5e, #e11d48);
}

/* Overlay Modal */
.overlay-modal-content {
  background: linear-gradient(180deg, #1e293b, #0f172a);
  padding: 28px;
  border-radius: 20px;
  width: 90%;
  max-width: 700px;
  text-align: center;
  border: 1px solid rgba(255, 255, 255, 0.1);
  box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
  animation: modalSlideUp 0.4s ease-out;
}

.overlay-modal-content h3 {
  color: #e2e8f0;
  font-size: 20px;
  font-weight: 600;
  margin: 0 0 24px 0;
}

.image-selection {
  display: flex;
  justify-content: center;
  gap: 16px;
  margin-bottom: 24px;
}

.thumbnail-container {
  display: flex;
  flex-direction: column;
  align-items: center;
  cursor: pointer;
  transition: all 0.3s ease;
  padding: 8px;
  border-radius: 12px;
}

.thumbnail-container:hover {
  transform: scale(1.05);
  background: rgba(255, 255, 255, 0.05);
}

.thumbnail-container.selected {
  background: rgba(59, 130, 246, 0.2);
  border: 2px solid var(--primary);
}

.thumbnail {
  width: 70px;
  height: 70px;
  object-fit: cover;
  border-radius: 10px;
  box-shadow: 0 4px 10px rgba(0, 0, 0, 0.2);
}

.thumbnail-container span {
  margin-top: 8px;
  font-size: 12px;
  color: #94a3b8;
}

.overlay-container {
  position: relative;
  width: 100%;
  height: 400px;
  margin-bottom: 20px;
  background: rgba(255, 255, 255, 0.05);
  border-radius: 16px;
  overflow: hidden;
  border: 1px solid rgba(255, 255, 255, 0.1);
}

.overlay-image {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  object-fit: contain;
}

.opacity-slider {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 12px;
  margin-bottom: 20px;
}

.opacity-slider label {
  color: #94a3b8;
  font-size: 14px;
}

.opacity-slider input[type="range"] {
  width: 200px;
  height: 6px;
  border-radius: 3px;
  background: rgba(255, 255, 255, 0.1);
  appearance: none;
  cursor: pointer;
}

.opacity-slider input[type="range"]::-webkit-slider-thumb {
  appearance: none;
  width: 18px;
  height: 18px;
  border-radius: 50%;
  background: var(--primary);
  box-shadow: 0 2px 8px rgba(59, 130, 246, 0.5);
  cursor: pointer;
}

.opacity-slider span {
  font-size: 14px;
  color: #60a5fa;
  font-weight: 600;
  min-width: 40px;
}

.close-btn {
  background: linear-gradient(135deg, var(--primary), var(--primary-dark));
  color: #fff;
  padding: 12px 32px;
  border-radius: 10px;
  border: none;
  cursor: pointer;
  font-weight: 600;
  font-size: 14px;
  transition: all 0.3s;
}

.close-btn:hover {
  transform: translateY(-2px);
  box-shadow: 0 8px 20px -5px rgba(59, 130, 246, 0.4);
}

/* AI Assistant Modal */
.ai-assistant-modal {
  position: fixed;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  width: 90%;
  max-width: 900px;
  max-height: 80vh;
  background: linear-gradient(180deg, #1e293b, #0f172a);
  box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
  border-radius: 20px;
  border: 1px solid rgba(255, 255, 255, 0.1);
  z-index: 102;
  padding: 0;
  overflow: hidden;
}

.modal-backdrop {
  position: fixed;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background-color: rgba(0, 0, 0, 0.7);
  backdrop-filter: blur(8px);
  z-index: 101;
}

/* Transitions */
.fade-enter-active,
.fade-leave-active {
  transition: opacity 0.3s ease;
}

.fade-enter-from,
.fade-leave-to {
  opacity: 0;
}

.ai-modal-enter-active,
.ai-modal-leave-active {
  transition: opacity 0.3s ease, transform 0.3s ease;
}

.ai-modal-enter-from {
  opacity: 0;
  transform: translate(-50%, -45%);
}

.ai-modal-enter-to {
  opacity: 1;
  transform: translate(-50%, -50%);
}

.ai-modal-leave-from {
  opacity: 1;
  transform: translate(-50%, -50%);
}

.ai-modal-leave-to {
  opacity: 0;
  transform: translate(-50%, -55%);
}

/* Responsive */
@media (max-width: 1200px) {
  .single-image .image-container {
    width: 450px;
    height: 450px;
  }
}

@media (max-width: 768px) {
  .container {
    padding: 20px;
  }
  
  .header-controls {
    flex-direction: column;
    gap: 16px;
  }
  
  .single-image .image-container {
    width: 100%;
    max-width: 400px;
    height: 350px;
  }
  
  .single-image .image-pair {
    flex-direction: column;
    align-items: center;
  }
  
  .color-legend {
    position: static;
    margin: 20px auto;
    max-width: 300px;
  }
  
  .horizontal-action-bar {
    bottom: 20px;
    right: 20px;
    gap: 12px;
  }
  
  .action-button {
    width: 60px;
    height: 60px;
  }
  
  .action-button span {
    font-size: 10px;
  }
}
</style>

.header-controls {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 30px;
}

.model-select select {
  padding: 10px 14px;
  border-radius: 8px;
  border: 1px solid #d2d2d7;
  background-color: #fff;
}

.btn {
  padding: 12px 24px;
  border-radius: 8px;
  border: none;
  cursor: pointer;
  transition: background-color 0.3s ease, transform 0.2s ease;
  font-family: 'WenQuanYi Micro Hei', sans-serif;
  font-weight: bold;
  color: #fff;
}

.btn:hover {
  transform: scale(1.05);
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
}

.batch-btn {
  background-color: rgb(118, 134, 146);
}

.image-display {
  display: flex;
  flex-direction: column;
  gap: 30px;
  align-items: center;
}

.single-image .image-pair {
  display: flex;
  justify-content: center;
  gap: 30px;
}

.image-container {
  background-color: #e8e8ed;
  border-radius: 16px;
  overflow: hidden;
  box-shadow: 0 6px 16px rgba(0, 0, 0, 0.1);
  display: flex;
  align-items: center;
  justify-content: center;
  position: relative;
  transition: transform 0.2s ease, box-shadow 0.2s ease;
}

.image-container:hover {
  transform: scale(1.02);
  box-shadow: 0 8px 20px rgba(0, 0, 0, 0.15);
}

.single-image .image-container {
  width: 600px;
  height: 600px;
}

.batch-images .image-container {
  width: 350px;
  height: 350px;
}

.image-container img {
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.image-row {
  display: flex;
  gap: 30px;
  justify-content: center;
  flex-wrap: wrap;
}

.zoom-btn {
  position: absolute;
  bottom: 10px;
  right: 10px;
  padding: 8px 16px;
  background-color: #007aff;
  color: #fff;
}

.modal {
  position: fixed;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background-color: rgba(0, 0, 0, 0.5);
  display: flex;
  justify-content: center;
  align-items: center;
}

.modal-content {
  background-color: #fff;
  padding: 20px;
  border-radius: 12px;
  display: flex;
  flex-direction: column;
  gap: 20px;
  width: 90%;
  max-width: 1000px;
  transform: translateY(-50px);
  animation: slideDown 0.3s ease forwards;
}

.modal-images {
  display: flex;
  gap: 20px;
  justify-content: center;
}

.modal-image-container {
  width: 400px;
  height: 400px;
  border-radius: 8px;
  overflow: hidden;
  background-color: #e8e8ed;
  display: flex;
  align-items: center;
  justify-content: center;
}

.modal-image-container img {
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.modal-stats {
  display: flex;
  gap: 20px;
  justify-content: center;
}

.color-legend {
  position: fixed;
  top: 20px;
  right: 20px;
  background-color: #fff;
  padding: 15px;
  border-radius: 12px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
}

.color-legend ul {
  list-style: none;
  padding: 0;
}

.color-legend li {
  display: flex;
  align-items: center;
  margin-bottom: 8px;
}

.color-box {
  width: 16px;
  height: 16px;
  margin-right: 8px;
  border-radius: 4px;
}

.loading {
  text-align: center;
  color: #86868b;
  margin-top: 20px;
}

.stats-container {
  display: flex;
  justify-content: center;
  align-items: flex-start;
  gap: 30px;
}

.chart-item {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 10px;
}

.export-btn {
  background-color: #28a745;
  color: #fff;
  padding: 8px 16px;
  border-radius: 8px;
  border: none;
  cursor: pointer;
}

.horizontal-action-bar {
  position: fixed;
  bottom: 30px;
  right: 30px;
  display: flex;
  gap: 20px;
  z-index: 100;
}

.action-button {
  width: 80px;
  height: 80px;
  border-radius: 50%;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 10px;
  color: #fff;
  border: none;
  cursor: pointer;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
  transition: transform 0.2s ease, box-shadow 0.2s ease;
  font-family: 'WenQuanYi Micro Hei', sans-serif;
  font-weight: bold;
  font-size: 16px;
}

.action-button:hover {
  transform: scale(1.05);
  box-shadow: 0 6px 16px rgba(0, 0, 0, 0.3);
}

.action-button .iconfont {
  font-size: 24px;
  margin-bottom: 5px;
}

.action-button span {
  font-size: 14px;
}

.ai-assistant-button {
  background-color: rgb(155, 158, 137);
}

.overlay-button {
  background-color: rgb(153, 164, 188);
}

.clear-button {
  background-color: rgb(167, 121, 121);
}

.overlay-modal-content {
  background: white;
  padding: 20px;
  border-radius: 8px;
  width: 600px;
  text-align: center;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
  animation: slideDown 0.3s ease;
}

.image-selection {
  display: flex;
  justify-content: center;
  gap: 20px;
  margin-bottom: 20px;
}

.thumbnail-container {
  display: flex;
  flex-direction: column;
  align-items: center;
  cursor: pointer;
  transition: transform 0.2s ease;
}

.thumbnail-container:hover {
  transform: scale(1.05);
}

.thumbnail-container.selected {
  border: 2px solid #007aff;
  border-radius: 8px;
}

.thumbnail {
  width: 80px;
  height: 80px;
  object-fit: cover;
  border-radius: 8px;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}

.thumbnail-container span {
  margin-top: 5px;
  font-size: 14px;
  color: #333;
}

.overlay-container {
  position: relative;
  width: 100%;
  height: 400px;
  margin-bottom: 20px;
  background-color: #e8e8ed;
  border-radius: 8px;
  overflow: hidden;
}

.overlay-image {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  object-fit: contain;
}

.opacity-slider {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 10px;
}

.opacity-slider input[type="range"] {
  width: 200px;
}

.opacity-slider span {
  font-size: 14px;
  color: #333;
}

.close-btn {
  background-color: #007aff;
  color: #fff;
  padding: 10px 20px;
  border-radius: 8px;
  border: none;
  cursor: pointer;
}

.close-btn:hover {
  opacity: 0.8;
}

@keyframes slideDown {
  from {
    transform: translateY(-50px);
  }
  to {
    transform: translateY(0);
  }
}

.ai-assistant-modal {
  position: fixed;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  width: 80%;
  max-width: 900px;
  max-height: 70vh;
  background-color: #fff;
  box-shadow: 0 8px 20px rgba(0, 0, 0, 0.2);
  border-radius: 15px;
  z-index: 102;
  padding: 20px;
  overflow: auto;
}

.modal-backdrop {
  position: fixed;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background-color: rgba(0, 0, 0, 0.5);
  z-index: 101;
}

.fade-enter-active,
.fade-leave-active {
  transition: opacity 0.3s ease;
}

.fade-enter-from,
.fade-leave-to {
  opacity: 0;
}

.fade-enter-to,
.fade-leave-from {
  opacity: 1;
}

.ai-modal-enter-active,
.ai-modal-leave-active {
  transition: opacity 0.3s ease, transform 0.3s ease;
}

.ai-modal-enter-from {
  opacity: 0;
  transform: translate(-50%, -60%);
}

.ai-modal-enter-to {
  opacity: 1;
  transform: translate(-50%, -50%);
}

.ai-modal-leave-from {
  opacity: 1;
  transform: translate(-50%, -50%);
}

.ai-modal-leave-to {
  opacity: 0;
  transform: translate(-50%, -40%);
}
</style>
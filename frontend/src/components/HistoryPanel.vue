<template>
  <div class="history-panel" :class="{ 'is-open': showHistory }">
    <button class="btn toggle-history-btn" @click="toggleHistory">
      {{ showHistory ? "收起历史" : "查看历史" }}
    </button>
    <div class="history-content" v-if="showHistory">
      <h3>历史记录</h3>
      <ul class="history-list">
        <li v-for="record in historyRecords" :key="record.image_id" @click="openHistoryModal(record)">
          <img :src="record.original_url" alt="原图" class="thumbnail" />
          <img :src="record.segmented_url" alt="分割图" class="thumbnail" />
          <span>{{ formatDate(record.created_at) }}</span>
          <button class="btn delete-btn" @click.stop="deleteHistoryItem(record.image_id)">删除</button>
        </li>
      </ul>
      <div class="pagination" v-if="totalHistory > perPage">
        <button class="btn" @click="changePage(-1)" :disabled="currentPage === 1">上一页</button>
        <span>{{ currentPage }} / {{ Math.ceil(totalHistory / perPage) }}</span>
        <button class="btn" @click="changePage(1)" :disabled="currentPage === Math.ceil(totalHistory / perPage)">下一页</button>
      </div>
      <button class="btn clear-history-btn" @click="clearHistory">清除历史记录</button>
    </div>
  </div>

  <!-- 历史记录详情模态框 -->
  <div v-if="showHistoryModal" class="modal" @click="closeHistoryModal">
    <div class="modal-content" @click.stop>
      <div class="modal-images">
        <div class="modal-image-container">
          <img :src="selectedHistory.original_url" alt="原始图像" />
        </div>
        <div class="modal-image-container">
          <img :src="selectedHistory.segmented_url" alt="分割图像" />
        </div>
      </div>
      <div class="modal-stats" v-if="selectedHistory && selectedHistory.stats">
        <div class="chart-item">
          <PieChart :chartData="selectedHistory.stats" :classColors="classColors" ref="historyPieChart" />
          <button class="btn export-btn" @click="exportChart('pie', 'history')">导出饼图</button>
        </div>
        <div class="chart-item">
          <HistogramChart :chartData="selectedHistory.stats" :classColors="classColors" ref="historyHistogramChart" />
          <button class="btn export-btn" @click="exportChart('histogram', 'history')">导出条形图</button>
        </div>
      </div>
      <button class="btn close-btn" @click="closeHistoryModal">关闭</button>
    </div>
  </div>
</template>

<script>
import axios from "axios";
import { ElMessage } from "element-plus";
import PieChart from "./PieChart.vue";
import HistogramChart from "./HistogramChart.vue";

export default {
  props: {
    classColors: { type: Object, required: true },
  },
  components: { PieChart, HistogramChart },
  data() {
    return {
      showHistory: false,
      historyRecords: [],
      currentPage: 1,
      perPage: 10,
      totalHistory: 0,
      showHistoryModal: false,
      selectedHistory: null,
    };
  },
  methods: {
    toggleHistory() {
      this.showHistory = !this.showHistory;
      if (this.showHistory) this.fetchHistory();
    },
    async fetchHistory() {
      const token = localStorage.getItem("token");
      try {
        const response = await axios.get(`http://localhost:8000/history?page=${this.currentPage}&per_page=${this.perPage}`, {
          headers: { Authorization: `Bearer ${token}` },
        });
        this.historyRecords = response.data.history.map(record => ({
          ...record,
          original_url: record.original_url.startsWith("http") ? record.original_url : `http://localhost:8000${record.original_url}`,
          segmented_url: record.segmented_url.startsWith("http") ? record.segmented_url : `http://localhost:8000${record.segmented_url}`,
        }));
        this.totalHistory = response.data.total;
      } catch (error) {
        ElMessage.error("获取历史记录失败");
        console.error(error);
      }
    },
    changePage(delta) {
      this.currentPage += delta;
      this.fetchHistory();
    },
    openHistoryModal(record) {
      this.selectedHistory = record;
      this.showHistoryModal = true;
    },
    closeHistoryModal() {
      this.showHistoryModal = false;
      this.selectedHistory = null;
    },
    exportChart(type, context) {
      let chartCanvas;
      if (type === "pie") {
        chartCanvas = this.$refs.historyPieChart.$refs.chartCanvas;
      } else if (type === "histogram") {
        chartCanvas = this.$refs.historyHistogramChart.$refs.chartCanvas;
      }
      if (chartCanvas) {
        const link = document.createElement("a");
        link.href = chartCanvas.toDataURL("image/png");
        link.download = `${type}_chart_${context === "history" ? this.selectedHistory.image_id : this.currentIndex || 0}.png`;
        link.click();
      }
    },
    formatDate(dateString) {
      const date = new Date(dateString);
      const year = date.getFullYear();
      const month = String(date.getMonth() + 1).padStart(2, "0");
      const day = String(date.getDate()).padStart(2, "0");
      const hours = String(date.getHours()).padStart(2, "0");
      const minutes = String(date.getMinutes()).padStart(2, "0");
      return `${year}-${month}-${day} ${hours}:${minutes}`;
    },
    async clearHistory() {
      const token = localStorage.getItem("token");
      try {
        await axios.delete("http://localhost:8000/clear_history", {
          headers: { Authorization: `Bearer ${token}` },
        });
        ElMessage.success("历史记录已清除");
        this.fetchHistory();
      } catch (error) {
        ElMessage.error("清除历史记录失败");
        console.error(error);
      }
    },
    async deleteHistoryItem(imageId) {
      const token = localStorage.getItem("token");
      try {
        await axios.delete(`http://localhost:8000/history/${imageId}`, {
          headers: { Authorization: `Bearer ${token}` },
        });
        ElMessage.success("历史记录已删除");
        this.fetchHistory();
      } catch (error) {
        ElMessage.error("删除历史记录失败");
        console.error(error);
      }
    },
  },
};
</script>

<style scoped>
.history-panel {
  position: fixed;
  top: 0;
  left: 0;
  height: 100%;
  width: 450px;
  background-color: #fff;
  box-shadow: 0 6px 16px rgba(0, 0, 0, 0.1);
  transform: translateX(-100%);
  transition: transform 0.3s ease;
  z-index: 1000;
}

.history-panel.is-open {
  transform: translateX(0);
}

.toggle-history-btn {
  position: absolute;
  top: 20px;
  right: -110px;
  background-color: rgb(118, 134, 146);
  color: #fff;
  padding: 10px 15px;
  border-radius: 8px;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
  cursor: pointer;
  z-index: 1001;
  font-family: 'WenQuanYi Micro Hei', sans-serif;
  font-weight: bold;
  transition: background-color 0.3s ease, transform 0.2s ease;
}

.toggle-history-btn:hover {
  transform: scale(1.05);
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
}

.history-content {
  padding: 20px;
  height: 100%;
  overflow-y: auto;
}

.history-list {
  list-style: none;
  padding: 0;
}

.history-list li {
  display: flex;
  align-items: center;
  gap: 15px;
  padding: 15px;
  border-bottom: 1px solid #e8e8ed;
  cursor: pointer;
}

.history-list li:hover {
  background-color: #f5f5f7;
}

.thumbnail {
  width: 120px;
  height: 120px;
  object-fit: cover;
  border-radius: 8px;
}

.pagination {
  display: flex;
  justify-content: center;
  align-items: center;
  gap: 10px;
  margin-top: 20px;
}

.clear-history-btn {
  background-color: rgb(167, 121, 121);
  color: #fff;
  margin-top: 20px;
  font-family: 'WenQuanYi Micro Hei', sans-serif;
  font-weight: bold;
  transition: background-color 0.3s ease, transform 0.2s ease;
}

.clear-history-btn:hover {
  transform: scale(1.05);
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
}

.delete-btn {
  background-color: rgb(167, 121, 121);
  color: #fff;
  padding: 5px 10px;
  border-radius: 4px;
  font-size: 14px;
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

.chart-item {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 10px;
}

.btn {
  padding: 10px 20px;
  border: none;
  border-radius: 8px;
  cursor: pointer;
  font-size: 16px;
  font-family: 'WenQuanYi Micro Hei', sans-serif;
  font-weight: bold;
  color: #fff;
  transition: background-color 0.3s ease, transform 0.2s ease;
}

.btn:hover {
  transform: scale(1.05);
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
}

.export-btn {
  background-color: #28a745;
  color: #fff;
}

.close-btn {
  background-color: #dc3545;
  color: #fff;
}
</style>
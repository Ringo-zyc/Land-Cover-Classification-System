<!-- PieChart.vue -->
<template>
  <div class="pie-chart">
    <canvas ref="chartCanvas" width="300" height="300"></canvas>
    <div v-if="!isDataValid" class="no-data">暂无数据</div>
  </div>
</template>

<script>
import Chart from "chart.js/auto";

export default {
  name: "PieChart",
  props: {
    chartData: { type: Object, required: true, default: () => ({}) },
    classColors: { type: Object, required: true },
  },
  data() {
    return { chart: null };
  },
  computed: {
    isDataValid() {
      if (!this.chartData || Object.keys(this.chartData).length === 0) return false;
      return Object.values(this.chartData).every((val) => typeof val === "number");
    },
  },
  watch: {
    chartData: {
      handler() { this.updateChart(); },
      deep: true,
    },
  },
  mounted() { if (this.isDataValid) this.updateChart(); },
  beforeUnmount() { if (this.chart) this.chart.destroy(); },
  methods: {
    updateChart() {
      const canvas = this.$refs.chartCanvas;
      if (!canvas || !this.isDataValid) return;

      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      if (this.chart) this.chart.destroy();

      const labels = Object.keys(this.chartData);
      const dataValues = Object.values(this.chartData);
      const backgroundColors = labels.map((label) => `rgba(${this.parseColorToRGB(this.classColors[label] || "#ccc")}, 1.0)`);
      const borderColors = labels.map((label) => (label === "Background" ? "#d2d2d7" : "transparent"));

      this.chart = new Chart(ctx, {
        type: "pie",
        data: {
          labels: labels,
          datasets: [{
            data: dataValues,
            backgroundColor: backgroundColors,
            borderColor: borderColors,
            borderWidth: 1,
          }],
        },
        options: {
          responsive: false,
          maintainAspectRatio: true,
          plugins: {
            legend: { display: false },
            tooltip: {
              enabled: true,
              callbacks: {
                label: (context) => {
                  const sum = context.dataset.data.reduce((a, b) => a + b, 0);
                  const percentage = Math.round((context.parsed / sum) * 100);
                  return `${context.label}: ${percentage}%`;
                },
              },
            },
            datalabels: { display: false },
          },
        },
      });
    },
    parseColorToRGB(color) {
      const rgb = color.match(/\d+/g);
      return rgb ? rgb.join(", ") : "255, 255, 255";
    },
  },
};
</script>

<style scoped>
.pie-chart {
  width: 300px;
  height: 300px;
  position: relative;
}

.no-data {
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  color: #86868b;
  font-size: 16px;
}
</style>
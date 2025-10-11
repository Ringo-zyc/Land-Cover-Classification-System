<template>
  <div class="histogram-chart">
    <canvas ref="chartCanvas" width="300" height="300"></canvas>
  </div>
</template>

<script>
import Chart from "chart.js/auto";
import ChartDataLabels from "chartjs-plugin-datalabels";

Chart.register(ChartDataLabels);

export default {
  name: "HistogramChart",
  props: {
    chartData: { type: Object, required: true, default: () => ({}) },
    classColors: { type: Object, required: true },
  },
  data() {
    return { chart: null };
  },
  computed: {
    totalSum() {
      return Object.values(this.chartData).reduce((sum, val) => sum + val, 0);
    },
    normalizedData() {
      if (this.totalSum === 0) return {};
      const data = {};
      for (const key in this.chartData) {
        data[key] = (this.chartData[key] / this.totalSum) * 100;
      }
      return data;
    },
  },
  watch: {
    chartData: {
      handler() { this.updateChart(); },
      deep: true,
    },
  },
  mounted() { if (this.totalSum > 0) this.updateChart(); },
  beforeUnmount() { if (this.chart) this.chart.destroy(); },
  methods: {
    updateChart() {
      const canvas = this.$refs.chartCanvas;
      if (!canvas || this.totalSum === 0) return;

      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      if (this.chart) this.chart.destroy();

      const labels = Object.keys(this.normalizedData);
      const dataValues = Object.values(this.normalizedData);
      const backgroundColors = labels.map((label) => this.classColors[label] || "#ccc");

      this.chart = new Chart(ctx, {
        type: "bar",
        data: {
          labels: labels,
          datasets: [{ label: "Percentage", data: dataValues, backgroundColor: backgroundColors, borderWidth: 1 }],
        },
        options: {
          responsive: false,
          maintainAspectRatio: true,
          scales: {
            y: { type: "logarithmic", beginAtZero: false, min: 0.01, title: { display: true, text: "百分比 (%)" } },
            x: { title: { display: true, text: "类别" } },
          },
          plugins: {
            legend: { display: false },
            tooltip: { callbacks: { label: (context) => `${context.parsed.y.toFixed(1)}%` } },
            datalabels: {
              anchor: (context) => (context.dataset.data[context.dataIndex] > 10 ? "center" : "end"),
              align: (context) => (context.dataset.data[context.dataIndex] > 10 ? "center" : "top"),
              formatter: (value) => `${value.toFixed(1)}%`,
              color: "#000",
              font: { weight: "bold" },
            },
          },
        },
      });
    },
  },
};
</script>

<style scoped>
.histogram-chart {
  width: 300px;
  height: 300px;
}
</style>
<template>
  <div class="benchmark-chart">
    <div class="chart-header">
      <h4>{{ title }}</h4>
      <div class="chart-legend">
        <span class="legend-item pytorch">PyTorch</span>
        <span class="legend-item triton">Triton (Ours)</span>
      </div>
    </div>

    <div class="chart-container">
      <div
        v-for="(item, index) in data"
        :key="index"
        class="chart-row"
      >
        <div class="row-label">{{ item.label }}</div>
        <div class="row-bars">
          <div
            class="bar pytorch-bar"
            :style="{ width: getWidth(item.pytorch) + '%' }"
          >
            <span class="bar-value">{{ item.pytorch }}</span>
          </div>
          <div
            class="bar triton-bar"
            :style="{ width: getWidth(item.triton) + '%' }"
          >
            <span class="bar-value">{{ item.triton }}</span>
            <span v-if="item.speedup" class="speedup-badge">{{ item.speedup }}x</span>
          </div>
        </div>
      </div>
    </div>

    <div class="chart-footer">
      <span class="metric-label">{{ yAxisLabel }}</span>
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue'

const props = defineProps({
  title: {
    type: String,
    default: 'Performance Comparison'
  },
  yAxisLabel: {
    type: String,
    default: 'Time (ms) ↓'
  },
  data: {
    type: Array,
    default: () => [
      { label: '512', pytorch: 0.8, triton: 0.7, speedup: '1.14' },
      { label: '1024', pytorch: 2.5, triton: 2.0, speedup: '1.25' },
      { label: '2048', pytorch: 9.0, triton: 6.5, speedup: '1.38' },
      { label: '4096', pytorch: 35.0, triton: 22.0, speedup: '1.59' },
    ]
  }
})

const maxValue = computed(() => {
  return Math.max(...props.data.map(d => Math.max(d.pytorch, d.triton)))
})

function getWidth(value) {
  return (value / maxValue.value) * 100
}
</script>

<style scoped>
.benchmark-chart {
  background: var(--vp-c-bg-soft);
  border: 1px solid var(--vp-c-divider);
  border-radius: 1rem;
  padding: 1.5rem;
  margin: 1.5rem 0;
}

.chart-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 1.5rem;
  flex-wrap: wrap;
  gap: 1rem;
}

.chart-header h4 {
  margin: 0;
  font-size: 1.1rem;
}

.chart-legend {
  display: flex;
  gap: 1rem;
  font-size: 0.875rem;
}

.legend-item {
  display: flex;
  align-items: center;
  gap: 0.375rem;
}

.legend-item::before {
  content: '';
  width: 12px;
  height: 12px;
  border-radius: 2px;
}

.legend-item.pytorch::before {
  background: #64748b;
}

.legend-item.triton::before {
  background: linear-gradient(135deg, #10b981, #059669);
}

.chart-container {
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

.chart-row {
  display: flex;
  align-items: center;
  gap: 1rem;
}

.row-label {
  width: 60px;
  font-weight: 600;
  font-size: 0.875rem;
  color: var(--vp-c-text-2);
  text-align: right;
}

.row-bars {
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: 0.375rem;
}

.bar {
  height: 28px;
  border-radius: 0.375rem;
  display: flex;
  align-items: center;
  justify-content: flex-end;
  padding-right: 0.75rem;
  font-size: 0.8rem;
  font-weight: 600;
  transition: width 1s ease-out;
  position: relative;
  animation: growBar 1s ease-out forwards;
  transform-origin: left;
}

@keyframes growBar {
  from { transform: scaleX(0); }
  to { transform: scaleX(1); }
}

.pytorch-bar {
  background: #64748b;
  color: white;
}

.triton-bar {
  background: linear-gradient(90deg, #10b981, #34d399);
  color: white;
}

.bar-value {
  margin-right: 0.5rem;
}

.speedup-badge {
  position: absolute;
  right: -50px;
  background: linear-gradient(135deg, #f59e0b, #fbbf24);
  color: white;
  padding: 0.125rem 0.5rem;
  border-radius: 9999px;
  font-size: 0.7rem;
  font-weight: 700;
  animation: popBadge 0.5s ease-out 0.5s both;
}

@keyframes popBadge {
  0% { transform: scale(0); opacity: 0; }
  80% { transform: scale(1.2); opacity: 1; }
  100% { transform: scale(1); opacity: 1; }
}

.chart-footer {
  margin-top: 1.5rem;
  text-align: right;
  font-size: 0.75rem;
  color: var(--vp-c-text-3);
  text-transform: uppercase;
  letter-spacing: 0.05em;
}
</style>

<template>
  <div class="gpu-visualizer">
    <div class="controls">
      <button
        v-for="arch in architectures"
        :key="arch.id"
        :class="['arch-btn', { active: selectedArch === arch.id }]"
        @click="selectArch(arch.id)"
      >
        {{ arch.name }}
      </button>
    </div>

    <div class="architecture-display">
      <Transition name="fade" mode="out-in">
        <div :key="selectedArch" class="arch-details">
          <div class="arch-header">
            <h3>{{ currentArch.name }}</h3>
            <span class="arch-badge">{{ currentArch.sm }}</span>
          </div>

          <div class="memory-hierarchy">
            <div
              v-for="(level, index) in currentArch.memory"
              :key="index"
              class="memory-level"
              :style="{ animationDelay: `${index * 100}ms` }"
            >
              <div class="memory-bar" :style="{ width: level.percentage + '%' }">
                <div class="memory-fill"></div>
              </div>
              <div class="memory-info">
                <span class="memory-name">{{ level.name }}</span>
                <span class="memory-value">{{ level.size }}</span>
                <span class="memory-speed">{{ level.speed }}</span>
              </div>
            </div>
          </div>

          <div class="features">
            <div
              v-for="feature in currentArch.features"
              :key="feature.name"
              class="feature-tag"
              :class="{ supported: feature.supported }"
            >
              <span class="feature-icon">{{ feature.supported ? '✓' : '×' }}</span>
              {{ feature.name }}
            </div>
          </div>
        </div>
      </Transition>
    </div>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'

const selectedArch = ref('ampere')

const architectures = [
  { id: 'volta', name: 'Volta', sm: 'SM70' },
  { id: 'turing', name: 'Turing', sm: 'SM75' },
  { id: 'ampere', name: 'Ampere', sm: 'SM80' },
  { id: 'ada', name: 'Ada', sm: 'SM89' },
  { id: 'hopper', name: 'Hopper', sm: 'SM90' },
]

const archData = {
  volta: {
    name: 'Volta (V100)',
    sm: 'SM70',
    memory: [
      { name: 'HBM', size: '16-32 GB', speed: '900 GB/s', percentage: 100 },
      { name: 'L2 Cache', size: '6 MB', speed: '~2 TB/s', percentage: 30 },
      { name: 'Shared Memory', size: '96 KB/SM', speed: '~15 TB/s', percentage: 5 },
    ],
    features: [
      { name: 'Tensor Cores', supported: true },
      { name: 'NVLink', supported: true },
      { name: 'Multi-Process Service', supported: true },
      { name: 'FP16', supported: true },
      { name: 'BF16', supported: false },
      { name: 'FP8', supported: false },
      { name: 'TMA', supported: false },
    ]
  },
  turing: {
    name: 'Turing (RTX 20xx)',
    sm: 'SM75',
    memory: [
      { name: 'GDDR6', size: '8-11 GB', speed: '616 GB/s', percentage: 85 },
      { name: 'L2 Cache', size: '4-6 MB', speed: '~2 TB/s', percentage: 25 },
      { name: 'Shared Memory', size: '64-96 KB/SM', speed: '~15 TB/s', percentage: 4 },
    ],
    features: [
      { name: 'Tensor Cores', supported: true },
      { name: 'RT Cores', supported: true },
      { name: 'NVLink', supported: true },
      { name: 'FP16', supported: true },
      { name: 'BF16', supported: false },
      { name: 'FP8', supported: false },
      { name: 'TMA', supported: false },
    ]
  },
  ampere: {
    name: 'Ampere (A100, RTX 30xx)',
    sm: 'SM80',
    memory: [
      { name: 'HBM2e/GDDR6X', size: '40-80 GB', speed: '2 TB/s', percentage: 100 },
      { name: 'L2 Cache', size: '40 MB', speed: '~4 TB/s', percentage: 50 },
      { name: 'Shared Memory', size: '164 KB/SM', speed: '~19 TB/s', percentage: 8 },
    ],
    features: [
      { name: '3rd Gen Tensor Cores', supported: true },
      { name: 'MIG', supported: true },
      { name: 'NVLink 3.0', supported: true },
      { name: 'FP16', supported: true },
      { name: 'BF16', supported: true },
      { name: 'TF32', supported: true },
      { name: 'FP8', supported: false },
      { name: 'TMA', supported: false },
    ]
  },
  ada: {
    name: 'Ada (RTX 40xx)',
    sm: 'SM89',
    memory: [
      { name: 'GDDR6X', size: '16-24 GB', speed: '1 TB/s', percentage: 90 },
      { name: 'L2 Cache', size: '48-72 MB', speed: '~4 TB/s', percentage: 55 },
      { name: 'Shared Memory', size: '192 KB/SM', speed: '~20 TB/s', percentage: 10 },
    ],
    features: [
      { name: '4th Gen Tensor Cores', supported: true },
      { name: 'DLSS 3', supported: true },
      { name: 'AV1 Encode', supported: true },
      { name: 'FP16', supported: true },
      { name: 'BF16', supported: true },
      { name: 'FP8', supported: false },
      { name: 'TMA', supported: false },
    ]
  },
  hopper: {
    name: 'Hopper (H100)',
    sm: 'SM90',
    memory: [
      { name: 'HBM3', size: '80 GB', speed: '3.35 TB/s', percentage: 100 },
      { name: 'L2 Cache', size: '50 MB', speed: '~6 TB/s', percentage: 60 },
      { name: 'Shared Memory', size: '228 KB/SM', speed: '~22 TB/s', percentage: 12 },
    ],
    features: [
      { name: '4th Gen Tensor Cores', supported: true },
      { name: 'Transformer Engine', supported: true },
      { name: 'DPX Instructions', supported: true },
      { name: 'NVLink 4.0', supported: true },
      { name: 'FP16', supported: true },
      { name: 'BF16', supported: true },
      { name: 'FP8', supported: true },
      { name: 'TMA', supported: true },
    ]
  }
}

const currentArch = computed(() => archData[selectedArch.value])

function selectArch(id) {
  selectedArch.value = id
}
</script>

<style scoped>
.gpu-visualizer {
  background: linear-gradient(135deg, rgba(16, 185, 129, 0.05) 0%, rgba(59, 130, 246, 0.05) 100%);
  border: 1px solid var(--vp-c-divider);
  border-radius: 1rem;
  padding: 1.5rem;
  margin: 1.5rem 0;
}

.controls {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
  margin-bottom: 1.5rem;
}

.arch-btn {
  padding: 0.5rem 1rem;
  border: 2px solid var(--vp-c-divider);
  border-radius: 9999px;
  background: transparent;
  color: var(--vp-c-text-2);
  font-weight: 500;
  cursor: pointer;
  transition: all 0.3s;
}

.arch-btn:hover {
  border-color: var(--vp-c-brand-1);
  color: var(--vp-c-brand-1);
}

.arch-btn.active {
  background: linear-gradient(135deg, var(--vp-c-brand-1), var(--vp-c-brand-2));
  border-color: transparent;
  color: white;
}

.arch-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 1.5rem;
}

.arch-header h3 {
  font-size: 1.5rem;
  font-weight: 700;
  margin: 0;
}

.arch-badge {
  padding: 0.25rem 0.75rem;
  background: var(--vp-c-brand-soft);
  color: var(--vp-c-brand-1);
  border-radius: 9999px;
  font-size: 0.875rem;
  font-weight: 600;
}

.memory-hierarchy {
  display: flex;
  flex-direction: column;
  gap: 1rem;
  margin-bottom: 1.5rem;
}

.memory-level {
  animation: slideIn 0.5s ease-out forwards;
  opacity: 0;
  transform: translateX(-20px);
}

@keyframes slideIn {
  to {
    opacity: 1;
    transform: translateX(0);
  }
}

.memory-bar {
  height: 8px;
  background: var(--vp-c-bg-mute);
  border-radius: 4px;
  overflow: hidden;
  margin-bottom: 0.5rem;
}

.memory-fill {
  height: 100%;
  background: linear-gradient(90deg, var(--vp-c-brand-1), var(--vp-c-brand-2));
  border-radius: 4px;
  animation: fillBar 1s ease-out forwards;
  transform-origin: left;
  transform: scaleX(0);
}

@keyframes fillBar {
  to {
    transform: scaleX(1);
  }
}

.memory-info {
  display: flex;
  justify-content: space-between;
  font-size: 0.875rem;
}

.memory-name {
  font-weight: 600;
  color: var(--vp-c-text-1);
}

.memory-value {
  color: var(--vp-c-brand-1);
  font-weight: 500;
}

.memory-speed {
  color: var(--vp-c-text-2);
}

.features {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
}

.feature-tag {
  display: flex;
  align-items: center;
  gap: 0.25rem;
  padding: 0.375rem 0.75rem;
  background: var(--vp-c-bg-mute);
  border-radius: 9999px;
  font-size: 0.75rem;
  color: var(--vp-c-text-2);
}

.feature-tag.supported {
  background: rgba(16, 185, 129, 0.1);
  color: var(--vp-c-brand-1);
}

.feature-icon {
  font-weight: 700;
}

.fade-enter-active,
.fade-leave-active {
  transition: opacity 0.3s, transform 0.3s;
}

.fade-enter-from,
.fade-leave-to {
  opacity: 0;
  transform: translateY(10px);
}
</style>

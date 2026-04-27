<template>
  <div class="flash-attention-viz">
    <div class="viz-header">
      <h4>FlashAttention Memory Flow</h4>
      <div class="legend">
        <span class="legend-item"><span class="dot hbm"></span> HBM</span>
        <span class="legend-item"><span class="dot sram"></span> SRAM</span>
        <span class="legend-item"><span class="arrow"></span> Data Flow</span>
      </div>
    </div>

    <div class="viz-container">
      <!-- HBM Layer -->
      <div class="memory-layer hbm-layer">
        <div class="layer-label">HBM (High Bandwidth Memory)</div>
        <div class="tensors">
          <div class="tensor q">Q<br><small>Query</small></div>
          <div class="tensor k">K<br><small>Key</small></div>
          <div class="tensor v">V<br><small>Value</small></div>
          <div class="tensor o">O<br><small>Output</small></div>
        </div>
      </div>

      <!-- Arrow Down -->
      <div class="flow-arrow down" :class="{ active: isAnimating }">
        <span class="arrow-label">Load Tiles</span>
      </div>

      <!-- SRAM Layer -->
      <div class="memory-layer sram-layer">
        <div class="layer-label">SRAM (Shared Memory - Fast!)</div>
        <div class="compute-blocks">
          <div
            v-for="(block, idx) in computeBlocks"
            :key="idx"
            class="compute-block"
            :class="{ active: currentBlock === idx }"
            :style="{ animationDelay: `${idx * 200}ms` }"
          >
            <div class="block-label">Block {{ idx + 1 }}</div>
            <div class="operations">
              <span :class="{ active: currentOp === 'load' }">Load</span>
              <span :class="{ active: currentOp === 'matmul' }">MatMul</span>
              <span :class="{ active: currentOp === 'softmax' }">Softmax</span>
              <span :class="{ active: currentOp === 'acc' }">Accumulate</span>
            </div>
          </div>
        </div>
      </div>

      <!-- Arrow Up -->
      <div class="flow-arrow up" :class="{ active: isAnimating }">
        <span class="arrow-label">Write Back</span>
      </div>

      <!-- Stats -->
      <div class="stats" v-if="showStats">
        <div class="stat">
          <div class="stat-value">{{ memorySaved }}%</div>
          <div class="stat-label">Memory Saved</div>
        </div>
        <div class="stat">
          <div class="stat-value">O(N)</div>
          <div class="stat-label">Complexity</div>
        </div>
        <div class="stat">
          <div class="stat-value">{{ hbmAccesses }}</div>
          <div class="stat-label">HBM Accesses</div>
        </div>
      </div>
    </div>

    <div class="viz-controls">
      <button @click="startAnimation" :disabled="isAnimating">
        {{ isAnimating ? 'Running...' : '▶ Start Animation' }}
      </button>
      <button @click="reset">Reset</button>
      <label>
        <input type="checkbox" v-model="showStats"> Show Stats
      </label>
    </div>

    <div class="explanation" v-if="currentStep">
      <strong>Step {{ currentStep.number }}:</strong> {{ currentStep.description }}
    </div>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'

const isAnimating = ref(false)
const showStats = ref(true)
const currentBlock = ref(-1)
const currentOp = ref('')
const currentStep = ref(null)

const memorySaved = computed(() => currentBlock.value >= 0 ? 99 : 0)
const hbmAccesses = computed(() => currentBlock.value >= 0 ? 'O(1)' : 'O(N²)')

const computeBlocks = [
  { name: 'Q₁K₁', ops: ['load', 'matmul', 'softmax'] },
  { name: 'Q₁K₂', ops: ['load', 'matmul', 'softmax', 'acc'] },
  { name: 'Q₂K₁', ops: ['load', 'matmul', 'softmax'] },
  { name: 'Q₂K₂', ops: ['load', 'matmul', 'softmax', 'acc'] },
]

const steps = [
  { number: 1, description: 'Load Q, K tiles from HBM to SRAM' },
  { number: 2, description: 'Compute attention scores Q×Kᵀ in SRAM' },
  { number: 3, description: 'Apply online softmax algorithm' },
  { number: 4, description: 'Multiply with V and accumulate result' },
  { number: 5, description: 'Write final output back to HBM' },
]

async function startAnimation() {
  if (isAnimating.value) return
  isAnimating.value = true

  for (let i = 0; i < computeBlocks.length; i++) {
    currentBlock.value = i
    currentStep.value = steps[0]
    await sleep(500)

    for (const op of computeBlocks[i].ops) {
      currentOp.value = op
      currentStep.value = steps[['load', 'matmul', 'softmax', 'acc'].indexOf(op) + 1] || steps[4]
      await sleep(400)
    }
  }

  currentOp.value = 'write'
  currentStep.value = steps[4]
  await sleep(500)

  isAnimating.value = false
  currentBlock.value = -1
  currentOp.value = ''
  currentStep.value = null
}

function reset() {
  isAnimating.value = false
  currentBlock.value = -1
  currentOp.value = ''
  currentStep.value = null
}

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms))
}
</script>

<style scoped>
.flash-attention-viz {
  background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
  border: 2px solid var(--vp-c-brand-1);
  border-radius: 1rem;
  padding: 1.5rem;
  margin: 1.5rem 0;
  color: white;
}

.viz-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 1.5rem;
  flex-wrap: wrap;
  gap: 1rem;
}

.viz-header h4 {
  margin: 0;
  color: var(--vp-c-brand-1);
}

.legend {
  display: flex;
  gap: 1rem;
  font-size: 0.75rem;
}

.legend-item {
  display: flex;
  align-items: center;
  gap: 0.25rem;
}

.dot {
  width: 12px;
  height: 12px;
  border-radius: 50%;
}

.dot.hbm { background: #3b82f6; }
.dot.sram { background: #10b981; }

.arrow::before {
  content: '↓';
  color: #f59e0b;
  font-weight: bold;
}

.viz-container {
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

.memory-layer {
  padding: 1rem;
  border-radius: 0.75rem;
  position: relative;
}

.hbm-layer {
  background: rgba(59, 130, 246, 0.1);
  border: 2px solid #3b82f6;
}

.sram-layer {
  background: rgba(16, 185, 129, 0.1);
  border: 2px solid #10b981;
}

.layer-label {
  font-size: 0.75rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  margin-bottom: 0.75rem;
  opacity: 0.8;
}

.tensors {
  display: flex;
  justify-content: space-around;
  gap: 0.5rem;
}

.tensor {
  width: 60px;
  height: 60px;
  background: linear-gradient(135deg, #3b82f6, #1d4ed8);
  border-radius: 0.5rem;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  font-weight: 700;
  font-size: 1.25rem;
  animation: pulse 2s infinite;
}

.tensor small {
  font-size: 0.6rem;
  font-weight: 400;
  opacity: 0.8;
}

@keyframes pulse {
  0%, 100% { transform: scale(1); }
  50% { transform: scale(1.05); }
}

.flow-arrow {
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 0.5rem;
  position: relative;
}

.flow-arrow::before {
  content: '';
  width: 2px;
  height: 30px;
  background: linear-gradient(to bottom, #f59e0b, #fbbf24);
  position: relative;
}

.flow-arrow.down::before {
  background: linear-gradient(to bottom, #3b82f6, #10b981);
}

.flow-arrow.up::before {
  background: linear-gradient(to bottom, #10b981, #3b82f6);
}

.flow-arrow.active::before {
  animation: flowAnimation 0.5s ease-in-out;
}

@keyframes flowAnimation {
  0% { opacity: 0.3; }
  50% { opacity: 1; height: 40px; }
  100% { opacity: 0.3; }
}

.arrow-label {
  position: absolute;
  font-size: 0.75rem;
  color: #f59e0b;
  font-weight: 600;
}

.flow-arrow.down .arrow-label {
  right: 0;
}

.flow-arrow.up .arrow-label {
  left: 0;
}

.compute-blocks {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(100px, 1fr));
  gap: 0.5rem;
}

.compute-block {
  background: rgba(255, 255, 255, 0.05);
  border: 1px solid rgba(255, 255, 255, 0.1);
  border-radius: 0.5rem;
  padding: 0.75rem;
  transition: all 0.3s;
}

.compute-block.active {
  background: rgba(16, 185, 129, 0.2);
  border-color: #10b981;
  box-shadow: 0 0 20px rgba(16, 185, 129, 0.3);
}

.block-label {
  font-size: 0.75rem;
  font-weight: 600;
  margin-bottom: 0.5rem;
  color: #10b981;
}

.operations {
  display: flex;
  flex-wrap: wrap;
  gap: 0.25rem;
}

.operations span {
  font-size: 0.65rem;
  padding: 0.125rem 0.375rem;
  background: rgba(255, 255, 255, 0.1);
  border-radius: 0.25rem;
  transition: all 0.2s;
}

.operations span.active {
  background: #10b981;
  color: white;
  font-weight: 600;
}

.stats {
  display: flex;
  justify-content: space-around;
  margin-top: 1rem;
  padding-top: 1rem;
  border-top: 1px solid rgba(255, 255, 255, 0.1);
}

.stat {
  text-align: center;
}

.stat-value {
  font-size: 1.5rem;
  font-weight: 800;
  color: #10b981;
}

.stat-label {
  font-size: 0.75rem;
  opacity: 0.7;
}

.viz-controls {
  display: flex;
  gap: 1rem;
  align-items: center;
  margin-top: 1.5rem;
  flex-wrap: wrap;
}

.viz-controls button {
  padding: 0.5rem 1rem;
  background: linear-gradient(135deg, #10b981, #059669);
  border: none;
  border-radius: 0.5rem;
  color: white;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.3s;
}

.viz-controls button:hover:not(:disabled) {
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(16, 185, 129, 0.4);
}

.viz-controls button:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.viz-controls label {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  font-size: 0.875rem;
  cursor: pointer;
}

.explanation {
  margin-top: 1rem;
  padding: 1rem;
  background: rgba(16, 185, 129, 0.1);
  border-radius: 0.5rem;
  font-size: 0.875rem;
  animation: fadeIn 0.3s;
}

@keyframes fadeIn {
  from { opacity: 0; transform: translateY(-10px); }
  to { opacity: 1; transform: translateY(0); }
}
</style>

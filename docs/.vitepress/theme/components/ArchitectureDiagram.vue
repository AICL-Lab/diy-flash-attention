<script setup lang="ts">
import { ref, computed } from 'vue'

// 视图类型
type ViewType = 'system' | 'memory' | 'dataflow'
const activeView = ref<ViewType>('system')

// 视图标签
const viewTabs: { key: ViewType; label: string; icon: string }[] = [
  { key: 'system', label: 'System Overview', icon: '🔷' },
  { key: 'memory', label: 'Memory Hierarchy', icon: '💾' },
  { key: 'dataflow', label: 'Data Flow', icon: '🔄' }
]
</script>

<template>
  <div class="architecture-diagram">
    <!-- 视图切换标签 -->
    <div class="view-tabs">
      <button
        v-for="tab in viewTabs"
        :key="tab.key"
        :class="['view-tab', { active: activeView === tab.key }]"
        @click="activeView = tab.key"
      >
        <span class="tab-icon">{{ tab.icon }}</span>
        <span class="tab-label">{{ tab.label }}</span>
      </button>
    </div>

    <!-- 视图内容 -->
    <div class="view-content">
      <!-- System Overview -->
      <div v-if="activeView === 'system'" class="diagram-container">
        <h3>FlashAttention System Architecture</h3>
        <pre class="mermaid">
flowchart TB
    subgraph Input["📥 Input Tensors"]
        Q["Q [B, H, N, D]"]
        K["K [B, H, N, D]"]
        V["V [B, H, N, D]"]
    end

    subgraph GPU["🖥️ GPU Processing"]
        subgraph Memory["Memory Hierarchy"]
            HBM["HBM (80GB)"]
            L2["L2 Cache (40MB)"]
            SRAM["SRAM/Shared Memory (228KB/SM)"]
        end

        subgraph Compute["Compute Units"]
            TC["Tensor Cores"]
            CUDA["CUDA Cores"]
        end
    end

    subgraph Kernel["⚡ FlashAttention Kernel"]
        direction TB
        Load["Load Q, K, V blocks to SRAM"]
        MM["Block MatMul (Tensor Cores)"]
        SM["Online Softmax"]
        Acc["Accumulate Output"]
        Store["Store O block to HBM"]
    end

    subgraph Output["📤 Output"]
        O["Output [B, H, N, D]"]
        L["Logsumexp [B, H, Nc]"]
    end

    Q --> Load
    K --> Load
    V --> Load

    HBM --> L2 --> SRAM
    SRAM --> Load

    Load --> MM --> SM --> Acc --> Store
    Acc -.-> TC
    MM -.-> TC

    Store --> O
    SM --> L

    style Kernel fill:#0e7490,stroke:#06b6d4,color:#fff
    style GPU fill:#1e293b,stroke:#475569,color:#fff
    style Input fill:#065f46,stroke:#10b981,color:#fff
    style Output fill:#7c3aed,stroke:#a78bfa,color:#fff
        </pre>

        <div class="legend">
          <div class="legend-item">
            <span class="legend-color" style="background: #0e7490;"></span>
            <span>FlashAttention Kernel</span>
          </div>
          <div class="legend-item">
            <span class="legend-color" style="background: #1e293b;"></span>
            <span>GPU Hardware</span>
          </div>
          <div class="legend-item">
            <span class="legend-color" style="background: #065f46;"></span>
            <span>Input Data</span>
          </div>
          <div class="legend-item">
            <span class="legend-color" style="background: #7c3aed;"></span>
            <span>Output Data</span>
          </div>
        </div>
      </div>

      <!-- Memory Hierarchy -->
      <div v-if="activeView === 'memory'" class="diagram-container">
        <h3>GPU Memory Hierarchy</h3>
        <pre class="mermaid">
flowchart LR
    subgraph Registers["⚡ Registers (Fastest)"]
        R1["32K per thread"]
        R2["~1 cycle latency"]
        R3["~20 TB/s bandwidth"]
    end

    subgraph SRAM["💾 Shared Memory / SRAM"]
        S1["228 KB per SM"]
        S2["~20 cycles latency"]
        S3["~19 TB/s bandwidth"]
    end

    subgraph L2["📦 L2 Cache"]
        L1["40-50 MB"]
        L2["~200 cycles latency"]
        L3["~4 TB/s bandwidth"]
    end

    subgraph HBM["📀 HBM (High Bandwidth Memory)"]
        H1["80 GB"]
        H2["~500 cycles latency"]
        H3["~3.35 TB/s bandwidth"]
    end

    Registers <--> SRAM <--> L2 <--> HBM

    subgraph FlashAttention["🎯 FlashAttention Optimization"]
        FA1["Load Q,K,V blocks to SRAM"]
        FA2["Compute in SRAM"]
        FA3["Only write O to HBM"]
        FA4["Avoid O(N²) HBM access"]
    end

    HBM --> FA1 --> SRAM
    SRAM --> FA2 --> FA3 --> HBM
    FA4 -.-> FA1

    style Registers fill:#dc2626,stroke:#ef4444,color:#fff
    style SRAM fill:#ea580c,stroke:#f97316,color:#fff
    style L2 fill:#ca8a04,stroke:#eab308,color:#fff
    style HBM fill:#16a34a,stroke:#22c55e,color:#fff
    style FlashAttention fill:#0e7490,stroke:#06b6d4,color:#fff
        </pre>

        <div class="memory-stats">
          <div class="stat-card">
            <div class="stat-value">99%</div>
            <div class="stat-label">Memory Saved<br>(Long Sequences)</div>
          </div>
          <div class="stat-card">
            <div class="stat-value">O(N)</div>
            <div class="stat-label">Attention Memory<br>Complexity</div>
          </div>
          <div class="stat-card">
            <div class="stat-value">4x</div>
            <div class="stat-label">Fewer HBM<br>Accesses</div>
          </div>
        </div>
      </div>

      <!-- Data Flow -->
      <div v-if="activeView === 'dataflow'" class="diagram-container">
        <h3>FlashAttention Data Flow</h3>
        <pre class="mermaid">
sequenceDiagram
    participant Host as CPU/Host
    participant HBM as HBM (GPU Memory)
    participant SRAM as SRAM (Shared Memory)
    participant TC as Tensor Cores
    participant Acc as Accumulator

    Note over Host,Acc: FlashAttention Forward Pass

    Host->>HBM: 1. Upload Q, K, V tensors [B, H, N, D]

    loop For each block (Br × Bc)
        HBM->>SRAM: 2. Load Q_i block [Br, D]
        HBM->>SRAM: 3. Load K_j block [Bc, D]
        HBM->>SRAM: 4. Load V_j block [Bc, D]

        SRAM->>TC: 5. Q_i @ K_j^T → S_ij
        Note right of TC: [Br, Bc] scores

        TC->>SRAM: 6. Store S_ij temporarily

        SRAM->>TC: 7. Online Softmax(S_ij)
        Note right of TC: m_ij, l_ij, P_ij

        TC->>Acc: 8. Update output: O_i += P_ij @ V_j
        Note right of Acc: Accumulate in SRAM

        SRAM->>TC: 9. Update logsumexp
    end

    Acc->>HBM: 10. Write output O [B, H, N, D]
    SRAM->>HBM: 11. Write logsumexp L [B, H, Nc]

    HBM->>Host: 12. Return results
        </pre>

        <div class="dataflow-notes">
          <div class="note-card">
            <h4>🔑 Key Innovation</h4>
            <p>Online Softmax allows computing attention in blocks without materializing the full N×N attention matrix.</p>
          </div>
          <div class="note-card">
            <h4>⚡ Memory Efficiency</h4>
            <p>Only Br×Bc intermediate results stored in SRAM, not N×N in HBM.</p>
          </div>
        </div>
      </div>
    </div>

    <!-- 底部说明 -->
    <div class="diagram-footer">
      <p>💡 Click on the tabs above to explore different views of the FlashAttention architecture.</p>
      <p>📖 For detailed explanations, see <a href="/diy-flash-attention/en/architecture">Architecture Design</a> and <a href="/diy-flash-attention/en/algorithm">Algorithm Deep Dive</a>.</p>
    </div>
  </div>
</template>

<style scoped>
.architecture-diagram {
  margin: 2rem 0;
  border-radius: 12px;
  background: var(--vp-c-bg-alt);
  border: 1px solid var(--vp-c-divider);
  overflow: hidden;
}

.view-tabs {
  display: flex;
  gap: 0.5rem;
  padding: 1rem;
  background: var(--vp-c-bg);
  border-bottom: 1px solid var(--vp-c-divider);
  overflow-x: auto;
}

.view-tab {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.75rem 1.25rem;
  border: 1px solid var(--vp-c-divider);
  border-radius: 8px;
  background: var(--vp-c-bg-soft);
  color: var(--vp-c-text-2);
  cursor: pointer;
  transition: all 0.2s ease;
  font-weight: 500;
  white-space: nowrap;
}

.view-tab:hover {
  border-color: var(--vp-c-brand-1);
  color: var(--vp-c-brand-1);
}

.view-tab.active {
  background: linear-gradient(135deg, var(--vp-c-brand-1), var(--vp-c-brand-2));
  border-color: var(--vp-c-brand-1);
  color: white;
}

.tab-icon {
  font-size: 1.1rem;
}

.tab-label {
  font-size: 0.9rem;
}

.view-content {
  padding: 1.5rem;
  min-height: 400px;
}

.diagram-container h3 {
  margin: 0 0 1rem 0;
  font-size: 1.25rem;
  font-weight: 600;
  color: var(--vp-c-text-1);
}

.diagram-container :deep(.mermaid) {
  background: var(--vp-c-bg);
  border-radius: 8px;
  padding: 1rem;
  margin: 1rem 0;
  overflow-x: auto;
}

.legend {
  display: flex;
  flex-wrap: wrap;
  gap: 1rem;
  margin-top: 1rem;
  padding: 1rem;
  background: var(--vp-c-bg);
  border-radius: 8px;
}

.legend-item {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  font-size: 0.85rem;
  color: var(--vp-c-text-2);
}

.legend-color {
  width: 12px;
  height: 12px;
  border-radius: 3px;
}

.memory-stats {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 1rem;
  margin-top: 1.5rem;
}

.stat-card {
  text-align: center;
  padding: 1.25rem;
  background: var(--vp-c-bg);
  border-radius: 8px;
  border: 1px solid var(--vp-c-divider);
}

.stat-value {
  font-size: 2rem;
  font-weight: 700;
  color: var(--vp-c-brand-1);
  line-height: 1.2;
}

.stat-label {
  font-size: 0.85rem;
  color: var(--vp-c-text-2);
  margin-top: 0.5rem;
}

.dataflow-notes {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 1rem;
  margin-top: 1.5rem;
}

.note-card {
  padding: 1rem;
  background: var(--vp-c-bg);
  border-radius: 8px;
  border-left: 3px solid var(--vp-c-brand-1);
}

.note-card h4 {
  margin: 0 0 0.5rem 0;
  font-size: 0.95rem;
  font-weight: 600;
  color: var(--vp-c-brand-1);
}

.note-card p {
  margin: 0;
  font-size: 0.85rem;
  color: var(--vp-c-text-2);
  line-height: 1.6;
}

.diagram-footer {
  padding: 1rem;
  background: var(--vp-c-bg);
  border-top: 1px solid var(--vp-c-divider);
  text-align: center;
}

.diagram-footer p {
  margin: 0.25rem 0;
  font-size: 0.85rem;
  color: var(--vp-c-text-3);
}

.diagram-footer a {
  color: var(--vp-c-brand-1);
  text-decoration: none;
}

.diagram-footer a:hover {
  text-decoration: underline;
}

@media (max-width: 768px) {
  .view-tabs {
    padding: 0.75rem;
  }

  .view-tab {
    padding: 0.5rem 0.75rem;
    font-size: 0.85rem;
  }

  .tab-label {
    display: none;
  }

  .tab-icon {
    font-size: 1.25rem;
  }

  .view-content {
    padding: 1rem;
  }
}
</style>

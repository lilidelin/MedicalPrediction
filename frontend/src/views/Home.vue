<template>
  <div class="home">
    <el-row :gutter="40">
      <el-col :span="12">
        <el-card class="welcome-card">
          <template #header>
            <div class="card-header">
              <h2>欢迎使用医学影像智能诊断系统</h2>
            </div>
          </template>
          
          <div class="welcome-content">
            <p>本系统基于深度学习技术，能够自动分析胸部X光片，快速识别肺炎症状。</p>
            
            <el-divider />
            
            <h3>系统特点：</h3>
            <ul>
              <li>🎯 <strong>高精度识别</strong>：基于ResNet架构的深度学习模型</li>
              <li>⚡ <strong>快速诊断</strong>：秒级完成图像分析</li>
              <li>🛡️ <strong>安全可靠</strong>：本地部署，保护患者隐私</li>
              <li>📊 <strong>详细报告</strong>：提供置信度和概率分布</li>
            </ul>
            
            <el-divider />
            
            <el-button type="primary" size="large" @click="$router.push('/upload')">
              <el-icon><Upload /></el-icon>
              开始诊断
            </el-button>
          </div>
        </el-card>
      </el-col>
      
      <el-col :span="12">
        <el-card class="stats-card">
          <template #header>
            <div class="card-header">
              <h2>系统状态</h2>
            </div>
          </template>
          
          <div class="stats-content">
            <el-row :gutter="20">
              <el-col :span="12">
                <div class="stat-item">
                  <div class="stat-value">{{ systemStatus.model_loaded ? '正常' : '异常' }}</div>
                  <div class="stat-label">模型状态</div>
                </div>
              </el-col>
              <el-col :span="12">
                <div class="stat-item">
                  <div class="stat-value">{{ systemStatus.device || 'CPU' }}</div>
                  <div class="stat-label">运行设备</div>
                </div>
              </el-col>
            </el-row>
            
            <el-divider />
            
            <div class="model-info" v-if="modelInfo">
              <h4>模型信息：</h4>
              <p><strong>模型名称：</strong>{{ modelInfo.model_name }}</p>
              <p><strong>类别数量：</strong>{{ modelInfo.num_classes }}</p>
              <p><strong>参数总量：</strong>{{ formatNumber(modelInfo.total_parameters) }}</p>
            </div>
            
            <el-button type="info" @click="refreshStatus" :loading="loading">
              <el-icon><Refresh /></el-icon>
              刷新状态
            </el-button>
          </div>
        </el-card>
      </el-col>
    </el-row>
  </div>
</template>

<script>
import { ref, onMounted } from 'vue'
import { ElMessage } from 'element-plus'
import axios from 'axios'

export default {
  name: 'Home',
  setup() {
    const systemStatus = ref({})
    const modelInfo = ref(null)
    const loading = ref(false)

    const API_BASE_URL = 'http://localhost:5000/api'

    const fetchSystemStatus = async () => {
      try {
        const response = await axios.get(`${API_BASE_URL}/health`)
        systemStatus.value = response.data
      } catch (error) {
        console.error('获取系统状态失败:', error)
        systemStatus.value = { status: 'error', model_loaded: false }
      }
    }

    const fetchModelInfo = async () => {
      try {
        const response = await axios.get(`${API_BASE_URL}/model-info`)
        modelInfo.value = response.data
      } catch (error) {
        console.error('获取模型信息失败:', error)
      }
    }

    const refreshStatus = async () => {
      loading.value = true
      await Promise.all([fetchSystemStatus(), fetchModelInfo()])
      loading.value = false
      ElMessage.success('状态已刷新')
    }

    const formatNumber = (num) => {
      if (num >= 1000000) {
        return (num / 1000000).toFixed(1) + 'M'
      } else if (num >= 1000) {
        return (num / 1000).toFixed(1) + 'K'
      }
      return num.toString()
    }

    onMounted(() => {
      fetchSystemStatus()
      fetchModelInfo()
    })

    return {
      systemStatus,
      modelInfo,
      loading,
      refreshStatus,
      formatNumber
    }
  }
}
</script>

<style scoped>
.home {
  max-width: 1200px;
  margin: 0 auto;
}

.welcome-card, .stats-card {
  height: 100%;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
}

.card-header h2 {
  margin: 0;
  color: #2c3e50;
}

.welcome-content {
  line-height: 1.6;
}

.welcome-content h3 {
  color: #409eff;
  margin-bottom: 15px;
}

.welcome-content ul {
  padding-left: 20px;
}

.welcome-content li {
  margin-bottom: 10px;
  color: #606266;
}

.stats-content {
  text-align: center;
}

.stat-item {
  padding: 20px;
  background: #f8f9fa;
  border-radius: 8px;
  margin-bottom: 20px;
}

.stat-value {
  font-size: 24px;
  font-weight: bold;
  color: #409eff;
  margin-bottom: 5px;
}

.stat-label {
  color: #909399;
  font-size: 14px;
}

.model-info {
  text-align: left;
  background: #f8f9fa;
  padding: 15px;
  border-radius: 8px;
  margin: 20px 0;
}

.model-info h4 {
  margin: 0 0 10px 0;
  color: #2c3e50;
}

.model-info p {
  margin: 5px 0;
  color: #606266;
}
</style> 
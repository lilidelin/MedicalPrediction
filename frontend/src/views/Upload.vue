<template>
  <div class="upload-page">
    <el-row :gutter="40">
      <el-col :span="12">
        <el-card class="upload-card">
          <template #header>
            <div class="card-header">
              <h2>图像上传</h2>
            </div>
          </template>
          
          <div class="upload-area">
            <el-upload
              ref="uploadRef"
              class="upload-demo"
              drag
              :auto-upload="false"
              :on-change="handleFileChange"
              :show-file-list="false"
              accept="image/*"
            >
              <el-icon class="el-icon--upload"><upload-filled /></el-icon>
              <div class="el-upload__text">
                将文件拖到此处，或<em>点击上传</em>
              </div>
              <template #tip>
                <div class="el-upload__tip">
                  支持 jpg/png 文件，且不超过 10MB
                </div>
              </template>
            </el-upload>
            
            <div v-if="selectedFile" class="selected-file">
              <el-alert
                :title="`已选择文件: ${selectedFile.name}`"
                type="success"
                :closable="false"
                show-icon
              />
            </div>
            
            <div class="upload-actions">
              <el-button 
                type="primary" 
                size="large" 
                @click="predictImage"
                :loading="predicting"
                :disabled="!selectedFile"
              >
                <el-icon><Search /></el-icon>
                开始诊断
              </el-button>
              
              <el-button 
                @click="clearSelection"
                :disabled="!selectedFile"
              >
                重新选择
              </el-button>
            </div>
          </div>
        </el-card>
      </el-col>
      
      <el-col :span="12">
        <el-card class="result-card">
          <template #header>
            <div class="card-header">
              <h2>诊断结果</h2>
            </div>
          </template>
          
          <div class="result-content">
            <div v-if="!predictionResult && !predicting" class="no-result">
              <el-empty description="请上传图像进行诊断" />
            </div>
            
            <div v-if="predicting" class="predicting">
              <el-progress type="circle" :percentage="progressPercentage" />
              <p>正在分析图像...</p>
            </div>
            
            <div v-if="predictionResult" class="prediction-result">
              <div class="result-header">
                <h3>诊断结论</h3>
                <el-tag 
                  :type="predictionResult.prediction === 'NORMAL' ? 'success' : 'danger'"
                  size="large"
                >
                  {{ predictionResult.prediction === 'NORMAL' ? '正常' : '肺炎' }}
                </el-tag>
              </div>
              
              <el-divider />
              
              <div class="confidence-section">
                <h4>置信度</h4>
                <el-progress 
                  :percentage="predictionResult.confidence" 
                  :color="getProgressColor(predictionResult.confidence)"
                  :stroke-width="20"
                />
                <p class="confidence-text">{{ predictionResult.confidence }}%</p>
              </div>
              
              <el-divider />
              
              <div class="probabilities-section">
                <h4>概率分布</h4>
                <div class="probability-bars">
                  <div class="probability-item">
                    <span class="label">正常</span>
                    <el-progress 
                      :percentage="predictionResult.probabilities.NORMAL"
                      color="#67C23A"
                      :stroke-width="15"
                    />
                    <span class="percentage">{{ predictionResult.probabilities.NORMAL }}%</span>
                  </div>
                  
                  <div class="probability-item">
                    <span class="label">肺炎</span>
                    <el-progress 
                      :percentage="predictionResult.probabilities.PNEUMONIA"
                      color="#F56C6C"
                      :stroke-width="15"
                    />
                    <span class="percentage">{{ predictionResult.probabilities.PNEUMONIA }}%</span>
                  </div>
                </div>
              </div>
              
              <el-divider />
              
              <div class="result-actions">
                <el-button type="primary" @click="downloadReport">
                  <el-icon><Download /></el-icon>
                  下载报告
                </el-button>
                <el-button @click="clearResult">
                  重新诊断
                </el-button>
              </div>
            </div>
          </div>
        </el-card>
      </el-col>
    </el-row>
  </div>
</template>

<script>
import { ref } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import axios from 'axios'

export default {
  name: 'Upload',
  setup() {
    const uploadRef = ref(null)
    const selectedFile = ref(null)
    const predicting = ref(false)
    const predictionResult = ref(null)
    const progressPercentage = ref(0)

    const API_BASE_URL = 'http://localhost:5000/api'

    const handleFileChange = (file) => {
      selectedFile.value = file.raw
      predictionResult.value = null
    }

    const clearSelection = () => {
      selectedFile.value = null
      predictionResult.value = null
      if (uploadRef.value) {
        uploadRef.value.clearFiles()
      }
    }

    const predictImage = async () => {
      if (!selectedFile.value) {
        ElMessage.warning('请先选择图像文件')
        return
      }

      predicting.value = true
      progressPercentage.value = 0

      // 模拟进度
      const progressInterval = setInterval(() => {
        if (progressPercentage.value < 90) {
          progressPercentage.value += 10
        }
      }, 200)

      try {
        const formData = new FormData()
        formData.append('file', selectedFile.value)

        const response = await axios.post(`${API_BASE_URL}/predict`, formData, {
          headers: {
            'Content-Type': 'multipart/form-data'
          }
        })

        clearInterval(progressInterval)
        progressPercentage.value = 100

        setTimeout(() => {
          predictionResult.value = response.data
          predicting.value = false
          ElMessage.success('诊断完成！')
        }, 500)

      } catch (error) {
        clearInterval(progressInterval)
        predicting.value = false
        console.error('预测失败:', error)
        
        ElMessage.error(error.response?.data?.error || '诊断失败，请重试')
      }
    }

    const clearResult = () => {
      predictionResult.value = null
    }

    const getProgressColor = (percentage) => {
      if (percentage >= 80) return '#67C23A'
      if (percentage >= 60) return '#E6A23C'
      return '#F56C6C'
    }

    const downloadReport = () => {
      if (!predictionResult.value) return

      const report = {
        timestamp: new Date().toLocaleString(),
        filename: selectedFile.value?.name || 'unknown',
        prediction: predictionResult.value.prediction,
        confidence: predictionResult.value.confidence,
        probabilities: predictionResult.value.probabilities
      }

      const blob = new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' })
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `diagnosis_report_${Date.now()}.json`
      document.body.appendChild(a)
      a.click()
      document.body.removeChild(a)
      URL.revokeObjectURL(url)

      ElMessage.success('报告已下载')
    }

    return {
      uploadRef,
      selectedFile,
      predicting,
      predictionResult,
      progressPercentage,
      handleFileChange,
      clearSelection,
      predictImage,
      clearResult,
      getProgressColor,
      downloadReport
    }
  }
}
</script>

<style scoped>
.upload-page {
  max-width: 1200px;
  margin: 0 auto;
}

.upload-card, .result-card {
  height: 100%;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
}

.card-header h2 {
  margin: 0;
  color: #2c3e50;
}

.upload-area {
  padding: 20px 0;
}

.upload-demo {
  width: 100%;
}

.selected-file {
  margin: 20px 0;
}

.upload-actions {
  margin-top: 20px;
  text-align: center;
}

.upload-actions .el-button {
  margin: 0 10px;
}

.result-content {
  min-height: 400px;
  display: flex;
  align-items: center;
  justify-content: center;
}

.no-result {
  text-align: center;
}

.predicting {
  text-align: center;
}

.predicting p {
  margin-top: 20px;
  color: #606266;
}

.prediction-result {
  width: 100%;
}

.result-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 20px;
}

.result-header h3 {
  margin: 0;
  color: #2c3e50;
}

.confidence-section {
  text-align: center;
}

.confidence-section h4 {
  margin-bottom: 15px;
  color: #2c3e50;
}

.confidence-text {
  margin-top: 10px;
  font-size: 18px;
  font-weight: bold;
  color: #409eff;
}

.probabilities-section h4 {
  margin-bottom: 15px;
  color: #2c3e50;
}

.probability-bars {
  display: flex;
  flex-direction: column;
  gap: 15px;
}

.probability-item {
  display: flex;
  align-items: center;
  gap: 15px;
}

.probability-item .label {
  min-width: 60px;
  font-weight: bold;
  color: #2c3e50;
}

.probability-item .el-progress {
  flex: 1;
}

.probability-item .percentage {
  min-width: 50px;
  text-align: right;
  font-weight: bold;
  color: #606266;
}

.result-actions {
  text-align: center;
  margin-top: 20px;
}

.result-actions .el-button {
  margin: 0 10px;
}
</style> 
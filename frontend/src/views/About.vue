<template>
  <div class="about-page">
    <el-row :gutter="40">
      <el-col :span="16">
        <el-card class="about-card">
          <template #header>
            <div class="card-header">
              <h2>关于系统</h2>
            </div>
          </template>
          
          <div class="about-content">
            <h3>系统简介</h3>
            <p>
              医学影像智能诊断系统是一个基于深度学习的医疗辅助诊断工具，专门用于胸部X光片的肺炎检测。
              该系统采用先进的计算机视觉技术，能够快速、准确地分析医学影像，为医生提供辅助诊断建议。
            </p>
            
            <h3>技术架构</h3>
            <div class="tech-stack">
              <div class="tech-item">
                <h4>前端技术</h4>
                <ul>
                  <li>Vue.js 3 - 现代化的前端框架</li>
                  <li>Element Plus - 企业级UI组件库</li>
                  <li>Axios - HTTP客户端</li>
                  <li>Vue Router - 单页面应用路由</li>
                </ul>
              </div>
              
              <div class="tech-item">
                <h4>后端技术</h4>
                <ul>
                  <li>Flask - Python Web框架</li>
                  <li>PyTorch - 深度学习框架</li>
                  <li>TorchVision - 计算机视觉库</li>
                  <li>PIL/Pillow - 图像处理库</li>
                </ul>
              </div>
              
              <div class="tech-item">
                <h4>模型架构</h4>
                <ul>
                  <li>ResNet - 残差神经网络</li>
                  <li>残差连接 - 解决梯度消失问题</li>
                  <li>批归一化 - 加速训练收敛</li>
                  <li>自适应池化 - 处理不同尺寸输入</li>
                </ul>
              </div>
            </div>
            
            <h3>功能特点</h3>
            <div class="features">
              <el-row :gutter="20">
                <el-col :span="12">
                  <div class="feature-item">
                    <el-icon class="feature-icon"><Cpu /></el-icon>
                    <h4>智能识别</h4>
                    <p>基于深度学习的自动图像分析，准确识别肺炎症状</p>
                  </div>
                </el-col>
                
                <el-col :span="12">
                  <div class="feature-item">
                    <el-icon class="feature-icon"><Timer /></el-icon>
                    <h4>快速诊断</h4>
                    <p>秒级完成图像分析，大幅提升诊断效率</p>
                  </div>
                </el-col>
                
                <el-col :span="12">
                  <div class="feature-item">
                    <el-icon class="feature-icon"><Shield /></el-icon>
                    <h4>安全可靠</h4>
                    <p>本地部署，保护患者隐私，确保数据安全</p>
                  </div>
                </el-col>
                
                <el-col :span="12">
                  <div class="feature-item">
                    <el-icon class="feature-icon"><DataAnalysis /></el-icon>
                    <h4>详细报告</h4>
                    <p>提供置信度和概率分布，支持诊断决策</p>
                  </div>
                </el-col>
              </el-row>
            </div>
            
            <h3>使用说明</h3>
            <div class="usage-steps">
              <el-steps :active="4" direction="vertical">
                <el-step title="上传图像" description="支持JPG、PNG等常见图像格式" />
                <el-step title="自动分析" description="系统自动进行图像预处理和特征提取" />
                <el-step title="智能诊断" description="基于深度学习模型进行肺炎检测" />
                <el-step title="查看结果" description="获得详细的诊断报告和置信度信息" />
              </el-steps>
            </div>
            
            <h3>注意事项</h3>
            <el-alert
              title="重要提醒"
              type="warning"
              :closable="false"
              show-icon
            >
              <template #default>
                <p>本系统仅供学习和研究使用，不能替代专业医生的诊断。</p>
                <p>实际医疗诊断请咨询专业医生。</p>
              </template>
            </el-alert>
          </div>
        </el-card>
      </el-col>
      
      <el-col :span="8">
        <el-card class="contact-card">
          <template #header>
            <div class="card-header">
              <h2>联系信息</h2>
            </div>
          </template>
          
          <div class="contact-content">
            <div class="contact-item">
              <el-icon><User /></el-icon>
              <div>
                <h4>开发者</h4>
                <p>医学影像AI研究团队</p>
              </div>
            </div>
            
            <div class="contact-item">
              <el-icon><Message /></el-icon>
              <div>
                <h4>邮箱</h4>
                <p>contact@medical-ai.com</p>
              </div>
            </div>
            
            <div class="contact-item">
              <el-icon><Location /></el-icon>
              <div>
                <h4>地址</h4>
                <p>中国·北京</p>
              </div>
            </div>
            
            <el-divider />
            
            <div class="version-info">
              <h4>版本信息</h4>
              <p><strong>当前版本：</strong>v1.0.0</p>
              <p><strong>更新时间：</strong>2024年1月</p>
              <p><strong>模型版本：</strong>SimpleResNet v1.0</p>
            </div>
          </div>
        </el-card>
      </el-col>
    </el-row>
  </div>
</template>

<script>
export default {
  name: 'About'
}
</script>

<style scoped>
.about-page {
  max-width: 1200px;
  margin: 0 auto;
}

.about-card, .contact-card {
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
}

.card-header h2 {
  margin: 0;
  color: #2c3e50;
}

.about-content {
  line-height: 1.6;
}

.about-content h3 {
  color: #409eff;
  margin: 30px 0 15px 0;
  border-bottom: 2px solid #f0f0f0;
  padding-bottom: 10px;
}

.about-content h3:first-child {
  margin-top: 0;
}

.tech-stack {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 20px;
  margin: 20px 0;
}

.tech-item {
  background: #f8f9fa;
  padding: 20px;
  border-radius: 8px;
  border-left: 4px solid #409eff;
}

.tech-item h4 {
  margin: 0 0 15px 0;
  color: #2c3e50;
}

.tech-item ul {
  margin: 0;
  padding-left: 20px;
}

.tech-item li {
  margin-bottom: 8px;
  color: #606266;
}

.features {
  margin: 20px 0;
}

.feature-item {
  text-align: center;
  padding: 20px;
  background: #f8f9fa;
  border-radius: 8px;
  margin-bottom: 20px;
}

.feature-icon {
  font-size: 40px;
  color: #409eff;
  margin-bottom: 15px;
}

.feature-item h4 {
  margin: 0 0 10px 0;
  color: #2c3e50;
}

.feature-item p {
  margin: 0;
  color: #606266;
  font-size: 14px;
}

.usage-steps {
  margin: 20px 0;
}

.contact-content {
  padding: 10px 0;
}

.contact-item {
  display: flex;
  align-items: center;
  margin-bottom: 20px;
  padding: 15px;
  background: #f8f9fa;
  border-radius: 8px;
}

.contact-item .el-icon {
  font-size: 24px;
  color: #409eff;
  margin-right: 15px;
}

.contact-item h4 {
  margin: 0 0 5px 0;
  color: #2c3e50;
}

.contact-item p {
  margin: 0;
  color: #606266;
}

.version-info {
  background: #f8f9fa;
  padding: 15px;
  border-radius: 8px;
}

.version-info h4 {
  margin: 0 0 15px 0;
  color: #2c3e50;
}

.version-info p {
  margin: 5px 0;
  color: #606266;
}
</style> 
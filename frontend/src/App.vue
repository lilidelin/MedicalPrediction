<template>
  <div id="app">
    <el-container>
      <el-header>
        <div class="header-content">
          <h1>🏥 医学影像智能诊断系统</h1>
          <p>基于深度学习的胸部X光片肺炎检测</p>
        </div>
      </el-header>
      
      <el-main>
        <router-view />
      </el-main>
      
      <el-footer>
        <p>&copy; 2024 医学影像智能诊断系统. 仅供学习和研究使用.</p>
      </el-footer>
    </el-container>
  </div>
</template>

<script>
export default {
  name: 'App'
}
</script>

<style>
#app {
  font-family: 'Helvetica Neue', Helvetica, 'PingFang SC', 'Hiragino Sans GB', 'Microsoft YaHei', '微软雅黑', Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  color: #2c3e50;
  min-height: 100vh;
}

.el-container {
  min-height: 100vh;
}

.el-header {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  text-align: center;
  padding: 20px 0;
}

.header-content h1 {
  margin: 0;
  font-size: 2.5em;
  font-weight: 300;
}

.header-content p {
  margin: 10px 0 0 0;
  font-size: 1.1em;
  opacity: 0.9;
}

.el-main {
  padding: 40px;
  background-color: #f5f7fa;
}

.el-footer {
  background-color: #2c3e50;
  color: white;
  text-align: center;
  padding: 20px;
}

.el-footer p {
  margin: 0;
  opacity: 0.8;
}
</style>

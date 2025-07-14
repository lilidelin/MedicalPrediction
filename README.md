# 医学影像智能诊断系统

基于深度学习的胸部X光片肺炎检测系统，采用Vue.js前端和Flask后端架构。

## 项目结构

```
MedicalPrediction/
├── backend/                 # Flask后端
│   ├── app.py              # 主应用文件
│   ├── model.py            # 模型定义
│   ├── requirements.txt    # Python依赖
│   └── model/              # 模型文件目录
│       └── best_model.pth  # 训练好的模型权重
├── frontend/               # Vue.js前端
│   ├── src/
│   │   ├── views/          # 页面组件
│   │   ├── router/         # 路由配置
│   │   ├── App.vue         # 根组件
│   │   └── main.js         # 入口文件
│   ├── public/             # 静态资源
│   ├── package.json        # 前端依赖
│   └── vue.config.js       # Vue配置
└── README.md               # 项目说明
```

## 功能特点

- 🎯 **高精度识别**：基于ResNet架构的深度学习模型
- ⚡ **快速诊断**：秒级完成图像分析
- 🛡️ **安全可靠**：本地部署，保护患者隐私
- 📊 **详细报告**：提供置信度和概率分布
- 🎨 **现代化UI**：基于Element Plus的美观界面

## 技术栈

### 后端
- **Flask** - Python Web框架
- **PyTorch** - 深度学习框架
- **TorchVision** - 计算机视觉库
- **PIL/Pillow** - 图像处理库

### 前端
- **Vue.js 3** - 现代化前端框架
- **Element Plus** - 企业级UI组件库
- **Axios** - HTTP客户端
- **Vue Router** - 单页面应用路由

### 模型
- **ResNet** - 残差神经网络
- **残差连接** - 解决梯度消失问题
- **批归一化** - 加速训练收敛

## 快速开始

### 1. 环境准备

确保你的系统已安装：
- Python 3.8+
- Node.js 14+
- npm 或 yarn

### 2. 后端设置

```bash
# 进入后端目录
cd backend

# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt

# 确保模型文件存在
# 将训练好的 best_model.pth 放到 backend/model/ 目录下

# 启动后端服务
python app.py
```

后端服务将在 `http://localhost:5000` 启动。

### 3. 前端设置

```bash
# 进入前端目录
cd frontend

# 安装依赖
npm install

# 启动开发服务器
npm run serve
```

前端应用将在 `http://localhost:8080` 启动。

### 4. 使用系统

1. 打开浏览器访问 `http://localhost:8080`
2. 点击"开始诊断"按钮
3. 上传胸部X光片图像
4. 查看诊断结果和置信度

## API接口

### 健康检查
```
GET /api/health
```

### 模型信息
```
GET /api/model-info
```

### 图像预测
```
POST /api/predict
Content-Type: multipart/form-data

参数：
- file: 图像文件
```

响应示例：
```json
{
  "prediction": "NORMAL",
  "confidence": 95.67,
  "probabilities": {
    "NORMAL": 95.67,
    "PNEUMONIA": 4.33
  }
}
```

## 开发说明

### 模型训练

模型使用以下配置训练：
- 数据集：胸部X光片数据集
- 模型：SimpleResNet
- 优化器：Adam
- 学习率：0.001
- 批次大小：32
- 训练轮数：20

### 自定义开发

1. **修改模型**：编辑 `backend/model.py`
2. **添加新接口**：在 `backend/app.py` 中添加路由
3. **修改前端**：编辑 `frontend/src/views/` 下的组件
4. **样式调整**：修改各组件中的 `<style>` 部分

## 部署说明

### 生产环境部署

1. **后端部署**：
   ```bash
   # 使用gunicorn
   pip install gunicorn
   gunicorn -w 4 -b 0.0.0.0:5000 app:app
   ```

2. **前端部署**：
   ```bash
   # 构建生产版本
   npm run build
   
   # 将dist目录部署到Web服务器
   ```

### Docker部署

```dockerfile
# 后端Dockerfile
FROM python:3.8-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 5000
CMD ["python", "app.py"]
```

## 注意事项

⚠️ **重要提醒**：
- 本系统仅供学习和研究使用
- 不能替代专业医生的诊断
- 实际医疗诊断请咨询专业医生
- 请确保模型文件路径正确

## 许可证

本项目仅供学习和研究使用。

## 联系方式

如有问题或建议，请联系开发团队。

---

**免责声明**：本系统仅用于学术研究和学习目的，不构成医疗建议。实际医疗诊断请咨询专业医生。

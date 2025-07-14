# backend/app.py
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from torchvision import transforms
from PIL import Image
import io
import base64
import numpy as np

# 导入模型结构
from model import SimpleResNet

app = Flask(__name__)
CORS(app)  # 允许跨域请求

# 全局变量
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None
transform = None

def load_model():
    """加载训练好的模型"""
    global model, transform
    
    # 初始化模型
    model = SimpleResNet(num_classes=2)
    
    # 加载模型权重
    model_path = os.path.join(os.path.dirname(__file__), 'model', 'best_model.pth')
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"模型已从 {model_path} 加载")
    else:
        print(f"警告：模型文件 {model_path} 不存在")
        return False
    
    model.to(device)
    model.eval()
    
    # 定义图像预处理
    channel_mean = [0.485, 0.456, 0.406]
    channel_std = [0.229, 0.224, 0.225]
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(channel_mean, channel_std)
    ])
    
    return True

def preprocess_image(image_data):
    """预处理图像数据"""
    try:
        # 如果是base64编码的图像
        if isinstance(image_data, str) and image_data.startswith('data:image'):
            # 移除data:image/jpeg;base64,前缀
            image_data = image_data.split(',')[1]
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
        else:
            # 如果是文件对象
            image = Image.open(image_data)
        
        # 转换为RGB格式
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # 应用预处理
        img_tensor = transform(image).unsqueeze(0)
        return img_tensor
        
    except Exception as e:
        print(f"图像预处理错误: {e}")
        return None

@app.route('/api/health', methods=['GET'])
def health_check():
    """健康检查接口"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'device': str(device)
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """预测接口"""
    try:
        if model is None:
            return jsonify({'error': '模型未加载'}), 500
        
        # 检查请求数据
        if 'file' in request.files:
            # 文件上传方式
            file = request.files['file']
            if file.filename == '':
                return jsonify({'error': '未选择文件'}), 400
            
            img_tensor = preprocess_image(file)
        elif 'image' in request.json:
            # Base64编码方式
            img_tensor = preprocess_image(request.json['image'])
        else:
            return jsonify({'error': '请提供图像文件或Base64编码的图像'}), 400
        
        if img_tensor is None:
            return jsonify({'error': '图像预处理失败'}), 400
        
        # 进行预测
        with torch.no_grad():
            img_tensor = img_tensor.to(device)
            outputs = model(img_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(outputs, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        # 标签映射
        label_map = {
            0: 'NORMAL',
            1: 'PNEUMONIA'
        }
        
        result = {
            'prediction': label_map[predicted_class],
            'confidence': round(confidence * 100, 2),
            'probabilities': {
                'NORMAL': round(probabilities[0][0].item() * 100, 2),
                'PNEUMONIA': round(probabilities[0][1].item() * 100, 2)
            }
        }
        
        return jsonify(result)
        
    except Exception as e:
        print(f"预测错误: {e}")
        return jsonify({'error': f'预测失败: {str(e)}'}), 500

@app.route('/api/model-info', methods=['GET'])
def model_info():
    """获取模型信息"""
    if model is None:
        return jsonify({'error': '模型未加载'}), 500
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return jsonify({
        'model_name': 'SimpleResNet',
        'num_classes': 2,
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'device': str(device),
        'classes': ['NORMAL', 'PNEUMONIA']
    })

if __name__ == '__main__':
    print("正在加载模型...")
    if load_model():
        print("模型加载成功！")
        print(f"使用设备: {device}")
        app.run(host='0.0.0.0', port=5000, debug=True)
    else:
        print("模型加载失败！请检查模型文件是否存在。")
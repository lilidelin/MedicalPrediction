<template>
    <div>
      <h2>医学影像智能诊断</h2>
      <input type="file" @change="onFileChange" accept="image/*" />
      <button @click="uploadImage" :disabled="!selectedFile">上传并预测</button>
      <div v-if="result">
        <h3>预测结果：{{ result }}</h3>
      </div>
    </div>
  </template>
  
  <script>
  import axios from 'axios'
  export default {
    data() {
      return {
        selectedFile: null,
        result: ''
      }
    },
    methods: {
      onFileChange(e) {
        this.selectedFile = e.target.files[0]
      },
      async uploadImage() {
        if (!this.selectedFile) return
        const formData = new FormData()
        formData.append('file', this.selectedFile)
        try {
          const res = await axios.post('http://localhost:5000/predict', formData, {
            headers: { 'Content-Type': 'multipart/form-data' }
          })
          this.result = res.data.result
        } catch (err) {
          this.result = '预测失败'
        }
      }
    }
  }
  </script>
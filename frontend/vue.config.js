const { defineConfig } = require('@vue/cli-service')

module.exports = defineConfig({
    transpileDependencies: true,
    devServer: {
        port: 8080,
        open: true,
        proxy: {
            '/api': {
                target: 'http://localhost:5000',
                changeOrigin: true,
                pathRewrite: {
                    '^/api': '/api'
                }
            }
        }
    },
    configureWebpack: {
        resolve: {
            alias: {
                '@': require('path').resolve(__dirname, 'src')
            }
        }
    }
})
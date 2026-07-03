"""
wsgi - Web 项目启动入口

Author: Grant Johnny
Version: 0.0.1
"""
from app import create_app

app = create_app()


@app.route('/')
def hello_world():
    return {'code': 0, 'message': 'Welcome to Heima!!!'}


if __name__ == '__main__':
    # 注意: 下面运行服务的方式仅供本地开发测试使用!
    # 生产环境请使用 WSGI 服务器 + Nginx 反向代理，切勿直接暴露到外网。
    #
    # Linux / macOS (使用 Gunicorn):
    #   gunicorn -w 4 -b 0.0.0.0:8080 --timeout 120 --graceful-timeout 30 --preload wsgi:app
    #
    # Windows (使用 Waitress，Gunicorn 不支持 Windows):
    #   waitress-serve --listen=0.0.0.0:8080 wsgi:app
    app.run(host='127.0.0.1', port=8080)

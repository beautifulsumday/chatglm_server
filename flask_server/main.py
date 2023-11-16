from flask import Flask, Response
import time

app = Flask(__name__)

def generate_data():
    """生成数据的生成器函数"""
    for i in range(10):
        yield f"data: {i}\n\n"
        time.sleep(1)  # 模拟数据生成的延迟

@app.route('/stream')
def stream():
    """流式响应的路由"""
    return Response(generate_data(), mimetype='text/event-stream')

if __name__ == '__main__':
    app.run(debug=True)
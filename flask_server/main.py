from flask import Flask, Response, request
import time

app = Flask(__name__)

def generate_data():
    """生成数据的生成器函数"""
    for i in range(10):
        yield f"data: {i}\n\n", f"data: {i}\n\n"
        time.sleep(1)  # 模拟数据生成的延迟

@app.route('/stream', methods=['GET', 'POST'])
def stream():
    """流式响应的路由"""
    data_json = request.json
    query = data_json.get("query")
    history = data_json.get("history")
    return Response(generate_data(), mimetype='text/event-stream')

if __name__ == '__main__':
    app.run(debug=True)
import requests

def stream_data(url):
    """从给定的 URL 流式接收数据"""
    with requests.get(url, stream=True) as response:
        for line in response.iter_lines():
            if line:
                print(line.decode('utf-8'))

if __name__ == '__main__':
    stream_url = 'http://127.0.0.1:5000/stream'
    stream_data(stream_url)

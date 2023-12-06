import requests

from kernel import CodeKernel, extract_code, execute
import json
TRUNCATE_LENGTH = 1024
kernel = CodeKernel()

def postprocess_text(text: str) -> str:
    text = text.replace("\(", "$")
    text = text.replace("\)", "$")
    text = text.replace("\[", "$$")
    text = text.replace("\]", "$$")
    text = text.replace("<|assistant|>", "")
    text = text.replace("<|observation|>", "")
    text = text.replace("<|system|>", "")
    text = text.replace("<|user|>", "")
    return text.strip()


def stream_code(url, data):
    again = False
    history = []
    with requests.post(url, stream=True, json=data) as response:
        for line in response.iter_content(decode_unicode=True, chunk_size=None): # 
            if line:
                if line.startswith("<|history|>"):
                    history =  line.split("<|history|>")[-1]
                    print("开始打印history")
                    print(history)
                    print("结束打印history")
                    history = json.loads(history)
                    # return
                elif line == "<|user|>":
                    """ 服务端代码
                    yield '<|user|>'
                    # 这里也应该返回一个http请求，表示llm结束了，其实就返回<|user|>就行，客户端那边接收到了之后就表示llm结束了
                    yield '<|history|>'+json.dumps(history)
                    return 
                    """
                    pass
                elif line == "<|assistant|>":
                    # 表示前端那边要换行了
                    print("\n")
                elif line.startswith("<|observation|>"):
                    code = line.split("<|observation|>")[-1]
                    print("这里是代码："+code)
                    try:
                        res_type, res = execute(code, kernel)
                    except Exception as e:
                        # st.error(f'Error when executing code: {e}')
                        return
                    
                    print("Received:", res_type, res)
                    if res_type == 'text' and len(res) > TRUNCATE_LENGTH:
                        res = res[:TRUNCATE_LENGTH] + ' [TRUNCATED]'

                    history.append({"role": "<|observation|>", 
                                    "content": '[Image]' if res_type == 'image' else postprocess_text(res)})

                    again = True
                else:
                    print(line, end="")
    if again:
        print("看一下这个history")
        print(history)
        stream_code(url, data={"history":history})


def stream_data(url):
    """从给定的 URL 流式接收数据"""
    data = {
        "query": "测试一下",
        "history": [{"a":1}, {"b":2}],

    }
    with requests.post(url, stream=True, json=data) as response:
        for line in response.iter_lines():
            if line:
                print(line.decode('utf-8'))


def test_chat(url):
    """测试对话"""
    data = {
        "query": "用pandas对该文件进行数据分析，至少可视化三个分析图案",
    }
    with requests.post(url, stream=True, json=data) as response:
        for line in response.iter_content(decode_unicode=True, chunk_size=1024):
            if line:
                print(line, end="")
                # print("====")
                # print("==================")
                # print(line.decode('gbk'))
                # print(line.decode('utf-8'))

if __name__ == '__main__':
    stream_url = 'http://127.0.0.1:5000/stream'
    stream_data(stream_url)
    # stream_url = 'http://127.0.0.1:50001/stream/chat'
    # test_chat(stream_url)

    # stream_url = 'http://127.0.0.1:6006/stream/interpreter2'
    # # test_chat(stream_url)

    # history = [{"role": "<|user|>", "content": "I uploaded the file and put it in /root/ChatGLM3/test.xlsx"}]
    # data = {
    #     "query": "对该文件进行数据分析，至少可视化三个分析图案",
    #     "history": history,
    # }

    # stream_code(stream_url, data)


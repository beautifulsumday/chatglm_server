import uuid
from transformers import AutoTokenizer, AutoModel
from huggingface_hub.inference._text_generation import TextGenerationStreamResponse, Token

from flask import Flask, Response, jsonify, request

from conversation import postprocess_text, preprocess_text, Conversation, Role
from typing import Any
import torch

from kernel import CodeKernel, extract_code, execute
import json
import time
import os

# SYSTEM_PROMPT = '你是一位智能AI助手，你叫ChatGLM，你连接着一台电脑，但请注意不能联网。在使用Python解决任务时，你可以运行代码并得到结果，如果运行结果有错误，你需要尽可能对代码进行改进。你可以处理用户上传到电脑上的文件，文件默认存储路径是/mnt/data/。'
SYSTEM_PROMPT = '你是一位智能AI助手，你叫DataGLM，你连接着用户的电脑，并且能访问本地文件！在使用Python解决用户的任务时，你可以准确的生成Python代码，当用户让你处理本地文件时，你能给用户提供准确代码并通过用户的代码运行结果进行下一步分析，如果返回运行结果有错误，你需要尽可能对代码进行改进，以完成用户的任务。'
# SYSTEM_PROMPT = '你是一位智能AI助手，你叫ChatGLM，你连接着一台电脑，并且可以联网。在使用Python解决任务时，你可以运行代码并得到结果，如果运行结果有错误，你需要尽可能对代码进行改进。如果有相关的包没有安装，你可以安装。你可以处理用户上传到电脑上的文件，文件默认存储路径是/mnt/data/。'
TOOL_PROMPT = 'Answer the following questions as best as you can. You have access to the following tools:\n'


MAX_LENGTH = 8192
TRUNCATE_LENGTH = 1024
temperature = 0.8
top_p = 0.8


model_path = "THUDM/chatglm3-6b"
model_path = "/root/autodl-tmp/model/model/"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
llm_model = AutoModel.from_pretrained(model_path, trust_remote_code=True).cuda()
# 多显卡支持，使用下面两行代替上面一行，将num_gpus改为你实际的显卡数量
# from utils import load_model_on_gpus
# model = load_model_on_gpus(model_path, num_gpus=2)
llm_model = llm_model.eval()

kernel = CodeKernel()

task_dic = {}

DATA_DIR = "/root/autodl-tmp/data/"

app = Flask(__name__)


def generate_data(query, history=None, past_key_values=None, return_past_key_values=True):
    # query = input("\n用户：") # 有什么功能,能为我做什么
    if history is None:
        history = []
    if query.strip() == "stop":
        return
    if query.strip() == "clear":
        past_key_values, history = None, []
        return
    
    current_length = 0
    for response in llm_model.stream_chat(tokenizer, query, history=history,
                                        past_key_values=past_key_values,
                                        return_past_key_values=return_past_key_values):
        if return_past_key_values:
            response, history, past_key_values = response
        else:
            response, history = response

        text = response[current_length:]
        current_length = len(response)
        if len(text)==0:
            continue
        yield text
    yield '<|history|>'
    yield json.dumps(history)
    

def stream_chat(model, tokenizer, query: str, history: list[tuple[str, str]] = None, role: str = "user",
                    past_key_values=None, max_length: int = 8192, do_sample=True, top_p=0.8, temperature=0.8,
                    logits_processor=None, return_past_key_values=False, **kwargs):
        
    from transformers.generation.logits_process import LogitsProcessor
    from transformers.generation.utils import LogitsProcessorList

    class InvalidScoreLogitsProcessor(LogitsProcessor):
        def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
            if torch.isnan(scores).any() or torch.isinf(scores).any():
                scores.zero_()
                scores[..., 5] = 5e4
            return scores

    if history is None:
        history = []
    if logits_processor is None:
        logits_processor = LogitsProcessorList()
    logits_processor.append(InvalidScoreLogitsProcessor())
    eos_token_id = [tokenizer.eos_token_id, tokenizer.get_command("<|user|>"),
                    tokenizer.get_command("<|observation|>")]
    gen_kwargs = {"max_length": max_length, "do_sample": do_sample, "top_p": top_p,
                    "temperature": temperature, "logits_processor": logits_processor, **kwargs}
    if past_key_values is None:
        inputs = tokenizer.build_chat_input(query, history=history, role=role)
    else:
        inputs = tokenizer.build_chat_input(query, role=role)
    inputs = inputs.to(model.device)
    if past_key_values is not None:
        past_length = past_key_values[0][0].shape[0]
        if model.transformer.pre_seq_len is not None:
            past_length -= model.transformer.pre_seq_len
        inputs.position_ids += past_length
        attention_mask = inputs.attention_mask
        attention_mask = torch.cat((attention_mask.new_ones(1, past_length), attention_mask), dim=1)
        inputs['attention_mask'] = attention_mask
    history.append({"role": role, "content": query})
    print("input_shape>", inputs['input_ids'].shape)

    input_sequence_length = inputs['input_ids'].shape[1]

    if max_length < input_sequence_length <= model.config.seq_length:
        yield "Current input sequence length {} exceeds sequence length set in generation parameters {}. The maximum model sequence length is {}. You may adjust the generation parameter to enable longer chat history.".format(
            input_sequence_length, max_length, model.config.seq_length
        ), history
        return

    if input_sequence_length > model.config.seq_length:
        yield "Current input sequence length {} exceeds maximum model sequence length {}. Unable to generate tokens.".format(
            input_sequence_length, model.config.seq_length
        ), history
        return

    for outputs in model.stream_generate(**inputs, past_key_values=past_key_values,
                                        eos_token_id=eos_token_id, return_past_key_values=return_past_key_values,
                                        **gen_kwargs):
        if return_past_key_values:
            outputs, past_key_values = outputs
        outputs = outputs.tolist()[0][len(inputs["input_ids"][0]):]
        response = tokenizer.decode(outputs)
        if response and response[-1] != "�":
            new_history = history
            if return_past_key_values:
                yield response, new_history, past_key_values
            else:
                yield response, new_history

def generate_stream(
        system: str | None,
        tools: list[dict] | None,
        history: list[dict],
        **parameters: Any
    ):

    chat_history = [{
        'role': 'system',
        'content': system if not tools else TOOL_PROMPT,
    }]

    if tools:
        chat_history[0]['tools'] = tools

    def transformer_role(role: str) -> str:
        match role:
            case "<|system|>":
                return "<|system|>"
            case "<|user|>":
                return "<|user|>"
            case "<|assistant|>" | "<|tool|>" | "<|interpreter|>":
                return "<|assistant|>"
            case "<|observation|>":
                return "<|observation|>"

    for conversation in history[:-1]:
        chat_history.append({
            'role': transformer_role(conversation.get("role")).removeprefix('<|').removesuffix('|>'),
            'content': conversation.get("content"),
        })

    # [{'role': 'system', 'content': '你是一位智能AI助手，你叫ChatGLM...mnt/data/。'}]
    query = history[-1].get("content")
    role = transformer_role(history[-1].get("role")).removeprefix('<|').removesuffix('|>')

    text = ''
    
    for new_text, _ in stream_chat(llm_model,
        tokenizer,
        query,
        chat_history,
        role,
        **parameters,
    ):
        word = new_text.removeprefix(text)
        word_stripped = word.strip()
        text = new_text
        yield TextGenerationStreamResponse(
            generated_text=text,
            token=Token(
                id=0,
                logprob=0,
                text=word,
                special=word_stripped.startswith('<|') and word_stripped.endswith('|>'),
            )
        )


# 规定，到达这里的history已经是[]格式的了，在这个函数里面通过文件加载history
def generate_with_id(task_id, query, history=None):
    global task_dic
    if history is None:
        history = []

    if query is not None:
        if query.strip() == "stop":
            return
        if query.strip() == "clear":
            if os.path.exists(os.path.join(DATA_DIR, task_id+".json")):
                with open(os.path.join(DATA_DIR, task_id+".json"), "w") as fw:
                    json.dump({"history": []}, fw)
            return
        
        query = query.strip()
        role = "<|user|>"
        # history.append({"role": role, "content": "I uploaded the file and put it in /root/ChatGLM3/test.xlsx"})
        history.append({"role": role, "content": query})
    else:
        if len(history) == 0:
            yield ""
            return
        else:
            # 如果有history，但是没有query，那就接下用llm生成下面的内容，比如收到来自远端的code运行结果等
            pass
    # 加载history json文件
    history_file = os.path.join(DATA_DIR, task_id+".json")
    with open(history_file, "r") as fr:
        old_history = json.load(fr)["history"]

    history = old_history + history

    print("history: ", history)
    for _ in range(5):
        output_text = ''
        for response in generate_stream(
            system=SYSTEM_PROMPT,
            tools=None,
            history=history,
            do_sample=True,
            max_length=MAX_LENGTH,
            temperature=temperature,
            top_p=top_p,
            stop_sequences=['<|user|>', '<|observation|>'],
        ): # stop_sequences = ['<|user|>', '<|observation|>']
            if task_dic[task_id] == "stop":
                task_dic[task_id] = "run"
                history.append({"role": "<|assistant|>", 
                            "content": postprocess_text(output_text)})
                with open(os.path.join(DATA_DIR, task_id+".json"), "w") as fw:
                                json.dump({"history": history}, fw)
                return
                
            token = response.token
            if response.token.special:

                match token.text.strip():
                    case '<|user|>':
                        history.append({"role": "<|assistant|>", "content": postprocess_text(output_text)})
                        yield '<|user|>'
                        # 这里也应该返回一个http请求，表示llm结束了，其实就返回<|user|>就行，客户端那边接收到了之后就表示llm结束了
                        with open(os.path.join(DATA_DIR, task_id+".json"), "w") as fw:
                            json.dump({"history": history}, fw)
                        # yield '<|history|>'+compress_string(json.dumps(history))
                        return
                    # Initiate tool call
                    case '<|assistant|>':
                        history.append({"role": "<|assistant|>", "content": postprocess_text(output_text)})
                        # 这里要输出到屏幕上，<|assistant|>表示表示llm要继续说话，这里客户那边应该换行
                        output_text = ''
                        yield '<|assistant|>'
                        continue
                    case '<|observation|>':
                        code = extract_code(output_text)
                        display_text = output_text.split('interpreter')[-1].strip()
                        history.append({"role": "<|interpreter|>", "content": postprocess_text(display_text)})
                        with open(os.path.join(DATA_DIR, task_id+".json"), "w") as fw:
                            json.dump({"history": history}, fw)
                        # yield '<|history|>'+compress_string(json.dumps(history))
                        time.sleep(1)
                        yield "<|observation|>"+code
                        return
                    case _:
                        # 这里应该返回一个http请求，表示llm错误，然后换行，继续llm开始生成内容
                        yield '<|unexpected_special_token|>'
                        # st.error(f'Unexpected special token: {token.text.strip()}')
                        with open(os.path.join(DATA_DIR, task_id+".json"), "w") as fw:
                            json.dump({"history": history}, fw)
                        # yield '<|history|>'+compress_string(json.dumps(history))
                        break

            yield response.token.text
            output_text += response.token.text  # 这里是+= 号，不是赋值=号，我说这个是为了防止你看错了
        else:
            history.append({"role": "<|assistant|>", 
                            "content": postprocess_text(output_text)})
            with open(os.path.join(DATA_DIR, task_id+".json"), "w") as fw:
                            json.dump({"history": history}, fw)
            # yield '<|history|>'+compress_string(json.dumps(history))
            return


@app.route('/stream/chat', methods=['GET', 'POST'])
def chat():
    data_json = request.json
    query = data_json.get("query")
    history = data_json.get("history")
    return Response(generate_data(query=query, history=history), mimetype='text/event-stream')

@app.route('/generate_task', methods=['GET'])
def generate_task():
    task_id = str(uuid.uuid4())
    session = {
        "history": []
    }
    with open(os.path.join(DATA_DIR, task_id+".json"), "w") as fw:
        json.dump(session, fw)

    global task_dic
    task_dic[task_id] = "run"

    return jsonify({"task_id": task_id})

@app.route('/stream/run', methods=['POST'])
def run():
    data_json = request.json
    task_id = data_json.get("task_id")
    query = data_json.get("query")
    history = data_json.get("history")

    if isinstance(history, str):
        history = json.loads(history)
    elif isinstance(history, list):
        pass
    else:
        history = []
    
    # 到这里，history已经是list格式了
    
    return Response(generate_with_id(task_id=task_id, query=query, history=history), mimetype='text/event-stream; charset=utf-8')


@app.route('/stop', methods=['POST'])
def stop():
    data_json = request.json
    task_id = data_json.get("task_id")

    global task_dic
    task_dic[task_id] = "stop"
    return jsonify({"status": "200", "message": "clear operation success"})


@app.route('/upload', methods=['POST'])
def upload():
    data_json = request.json
    task_id = data_json.get("task_id")
    upload_file = data_json.get("upload_file")


    if (upload_file is not None) and (task_id is not None):
        # 加载history json文件
        history_file = os.path.join(DATA_DIR, task_id+".json")
        with open(history_file, "r") as fr:
            old_history = json.load(fr)["history"]
        
        old_history.append({'role': '<|user|>', 'content': 'I uploaded the file and put it in '+upload_file})
        # 保存history json文件
        with open(os.path.join(DATA_DIR, task_id+".json"), "w") as fw:
            json.dump({"history": old_history}, fw)

        return jsonify({"status": "200", "message": "upload operation success"})
    else:
        return jsonify({"status": "400", "message": "task id or upload file is none"})



if __name__ == '__main__':
    app.run(debug=False, port=6006)
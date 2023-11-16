import streamlit as st
from streamlit.delta_generator import DeltaGenerator

from client import get_client
from conversation import postprocess_text, preprocess_text, Conversation, Role

MAX_LENGTH = 8192

client = get_client()

# Append a conversation into history, while show it in a new markdown block
def append_conversation(
    conversation: Conversation,
    history: list[Conversation],
    placeholder: DeltaGenerator | None=None,
) -> None:
    history.append(conversation)
    # 屏幕上输出conversation的内容
    conversation.show(placeholder)

def main(top_p: float, temperature: float, system_prompt: str, prompt_text: str):
    placeholder = st.empty()
    with placeholder.container():
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []

        # history存储了所有的对话内容
        history: list[Conversation] = st.session_state.chat_history

        # streamlit每次query都是一个http请求，所以要history要保存之前的query和回答
        # 并把以前的对话内容重新打印到屏幕上
        for conversation in history:
            conversation.show()

    if prompt_text:
        prompt_text = prompt_text.strip()
        # append_conversation做的动作就是把对话内容加入到history中，并添加到streamlit的placeholder
        # 到时候就可以在streamlit的placeholder中显示对话内容
        append_conversation(Conversation(Role.USER, prompt_text), history)

        # 由于上一步把prompt_text添加到了history中，所以这里的preprocess_text输入history就行
        # 把历史的所有对话内容都拼接起来，然后再加上system_prompt，并加上tool，一起输入，然后在回答下一句话
        # 
        input_text = preprocess_text(
            system_prompt,
            tools=None,
            history=history,
        )
        print("=== Input:")
        print(input_text)
        print("=== History:")
        print(history)

        placeholder = st.empty()
        message_placeholder = placeholder.chat_message(name="assistant", avatar="assistant")
        markdown_placeholder = message_placeholder.empty()

        output_text = ''
        for response in client.generate_stream(
            system_prompt,
            tools=None, 
            history=history,
            do_sample=True,
            max_length=MAX_LENGTH,
            temperature=temperature,
            top_p=top_p,
            stop_sequences=[str(Role.USER)],
        ):
            token = response.token
            if response.token.special:
                print("=== Output:")
                print(output_text)

                match token.text.strip():
                    case '<|user|>':
                        break
                    case _:
                        st.error(f'Unexpected special token: {token.text.strip()}')
                        break
            output_text += response.token.text
            # 这里是屏幕一个一个的蹦出字来
            markdown_placeholder.markdown(postprocess_text(output_text + '▌'))
        
        append_conversation(Conversation(
            Role.ASSISTANT,
            postprocess_text(output_text),
        ), history, markdown_placeholder)
# from modules.ui import create_ui
from argparse import ArgumentParser

import gradio as gr
import os, json, types
from typing import Optional, List, Tuple
import copy
os.environ["RWKV_JIT_ON"] = '0'
from src.model_run import RWKV_RNN
import numpy as np
import os, copy, types, gc, sys
import torch
from src.utils import TOKENIZER
  # '1' or '0', please use torch 1.13+ and benchmark speed
from rwkv.utils import PIPELINE, PIPELINE_ARGS

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
np.set_printoptions(precision=4, suppress=True, linewidth=200)

parser = ArgumentParser()

parser.add_argument("--MODEL_NAME", default='', type=str, help="基础模型的地址，去掉.pth")  # full path, with .pth
parser.add_argument("--MODEL_LORA", default='', type=str, help="lora模型的地址，去掉.pth")  # full path, with .pth
parser.add_argument("--n_layer", default=32, type=int, help="lora模型的地址，去掉.pth")  # full path, with .pth
parser.add_argument("--n_embd", default=4096, type=int)  # full path, with .pth
parser.add_argument("--ctx_len", default=4096, type=int)  # full path, with .pth
parser.add_argument("--lora_r", default=16, type=int, help='Modify this to use LoRA models; lora_r = 0 will not use LoRA weights.')  # full path, with .pth
parser.add_argument("--lora_alpha", default=32, type=int)  # full path, with .pth

args = parser.parse_args()
CHAT_LANG = 'Chinese' # English Chinese

WORD_NAME = [
    "20B_tokenizer.json",
    "20B_tokenizer.json",
]  # [vocab, vocab] for Pile model
UNKNOWN_CHAR = None
tokenizer = TOKENIZER(WORD_NAME, UNKNOWN_CHAR=UNKNOWN_CHAR)

args.RUN_DEVICE = "cuda"  # 'cpu' (already very fast) // 'cuda'
args.FLOAT_MODE = "fp16" # fp32 (good for CPU) // fp16 (recommended for GPU) // bf16 (less accurate)
args.vocab_size = 50277
args.head_qk = 0
args.pre_ffn = 0
args.grad_cp = 0
args.my_pos_emb = 0

args.n_layer = 32
args.n_embd = 4096
args.ctx_len = 4096

# Modify this to use LoRA models; lora_r = 0 will not use LoRA weights.
args.lora_r = 16
args.lora_alpha = 32


user = "Q"
bot = "A"
interface = ":"

os.environ["RWKV_RUN_DEVICE"] = args.RUN_DEVICE
MODEL_NAME = args.MODEL_NAME
print(f'loading... {MODEL_NAME}')
model = RWKV_RNN(args)

model_tokens = []

current_state = None

########################################################################################################
init_prompt = "初始prompt"



css = "style.css"
# current_path = os.path.dirname(os.path.abspath(__file__))
# CHUNK_LEN = 256
# AVOID_REPEAT = '，：？！'
#
# model = RWKV(model=args.MODEL_NAME, strategy=args.strategy)
# pipeline = PIPELINE(model, f"{current_path}/20B_tokenizer.json")
# END_OF_TEXT = 0
# END_OF_LINE = 187
# END_OF_LINE_2 = 535
# CHAT_LEN_SHORT = 40
# CHAT_LEN_LONG = 150
# # English Chinese
# PROMPT_FILE = f'{current_path}/prompt/English-2.py'
# # PROMPT_FILE = f'{current_path}/prompt/Chinese-2.py'
# alpha_presence = 0.5
# alpha_frequency = 0.5
# with open(PROMPT_FILE, 'rb') as file:
#     user = None
#     bot = None
#     interface = None
#     init_prompt = None
#     exec(compile(file.read(), PROMPT_FILE, 'exec'))
# init_prompt = init_prompt.strip().split('\n')
# for c in range(len(init_prompt)):
#     init_prompt[c] = init_prompt[c].strip().strip('\u3000').strip('\r')
# init_prompt = '\n' + ('\n'.join(init_prompt)).strip() + '\n\n'
#
# AVOID_REPEAT_TOKENS = []
# for i in AVOID_REPEAT:
#     dd = pipeline.encode(i)
#     assert len(dd) == 1
#     AVOID_REPEAT_TOKENS += dd
# if not os.path.isfile('config.json'):
#     save_config()
#
# CHAR_STOP = '\n\n'


class Context:
    def __init__(self, model_state=None, model_state_init=None, model_state_gen_0=None,
                 history: Optional[List[Tuple[str, str]]] = None):
        if history != None:
            self.history = history
        else:
            self.history = []
        if model_state_init != None:
            self.model_state_init = copy.deepcopy(model_state_init)
        else:
            self.model_state_init = None
        if model_state != None:
            self.model_state = copy.deepcopy(model_state)
        else:
            self.model_state = copy.deepcopy(model_state_init)
        self.model_state_gen_0 = copy.deepcopy(model_state_init)

    def append(self, query, output):
        self.history.append((query, output))

    def clear_history(self):
        self.history = []
        self.model_state = copy.deepcopy(self.init_model_state)

    def clear_all_history(self):
        self.history = []
        self.model_state = None
        self.init_model_state = None

    def load_last_state(self):
        self.model_state = copy.deepcopy(self.model_state_gen_0)


# def run_rnn(model_state, tokens, newline_adj=0):
#     for i in range(len(tokens)):
#         tokens += [int(tokens[i])]
#         if i == len(tokens) - 1:
#             out, model_state = model.forward(tokens, model_state)
#         else:
#             model_state = model.forward(tokens, model_state, preprocess_only=True)
#
#     out[0] = -999999999
#     out[187] += newline_adj  # adjust \n probability
#
#
#     return out, model_state


def read_tokens(tokens, model_state, chunk_len=256):
    model_tokens = []
    for i in range(len(tokens)):
        model_tokens += [int(tokens[i])]
        if i == len(tokens) - 1:
            out, model_state = model.forward(model_tokens, model_state)
        else:
            model_state = model.forward(model_tokens, model_state, preprocess_only=True)
    out[0] = -999999999
    return out, model_state


def out_prob_adj_chat(
        out,
        occurrence,
        alpha_presence,
        alpha_frequency):
    for n in occurrence:
        out[n] -= (alpha_presence + occurrence[n] * alpha_frequency)
    return out


def out_prob_adj_gen(
        out,
        occurrence,
        alpha_presence,
        alpha_frequency):
    for n in occurrence:
        out[n] -= (alpha_presence + occurrence[n] * alpha_frequency)
    return out


def stream_generate_from_out(out, model_state, out_prob_adj=out_prob_adj_gen, token_count=100, args=PIPELINE_ARGS()):
    out_tokens = []
    model_tokens = [16]
    out_last = 0
    occurrence = {}
    i = 0
    out_str = ""
    while (token_count < 0) | (i < token_count):
        # out = out_prob_adj(out, occurrence, alpha_presence, alpha_frequency)
        token = tokenizer.sample_logits(
            out,
            model_tokens,
            ctx_len=4096,
            temperature=args.temperature,
            top_p_usual=args.top_p,
            top_p_newline=args.top_p,
        )
        if token == '\n\n':
            break
        if token not in occurrence:
            occurrence[token] = 1
        else:
            occurrence[token] += 1
        out_tokens += [token]
        model_tokens += [token]
        out, model_state = model.forward(model_tokens, model_state)
        tmp = tokenizer.tokenizer.decode(out_tokens[out_last:])
        if '\ufffd' not in tmp:  # avoid utf-8 display issues
            if "\n\n" in tmp:
                break
            out_str += tmp
            out_last = i + 1
            yield out_str, model_state
        i = i + 1
    yield out_str, model_state


def chat(query, history, model_state, max_length, top_p, temperature):
    # history = history + [(query,None)]
    # ctx.history[-1][1]=""
    # query=ctx.history[-1][0]
    history = history + [(query, "")]
    query_text = f"{user}{interface} {query}\n\n{bot}{interface} "
    tokens = tokenizer.tokenizer.encode(query_text)
    out, model_state = read_tokens(tokens, model_state)
    last_model_state = copy.deepcopy(model_state)
    last_out = copy.deepcopy(out)
    for response, model_state in stream_generate_from_out(out, model_state, out_prob_adj=out_prob_adj_chat,
                                                          token_count=max_length, args=PIPELINE_ARGS(top_p=top_p, temperature=temperature)):
        history[-1] = (query, response)
        if "\n\n" in response:
            break
        yield "", history, model_state, last_out, last_model_state


def regen_last(history, last_out, last_model_state, max_length, top_p, temperature):
    if len(history) == 0:
        yield history, last_model_state
    for response, model_state in stream_generate_from_out(last_out, last_model_state, out_prob_adj=out_prob_adj_chat,
                                                          token_count=-1, args=PIPELINE_ARGS(top_p=top_p, temperature=temperature)):
        history[-1][1] = response
        yield history, model_state


def parse_text(text):
    lines = text.split('\n')
    for i, line in enumerate(lines):
        if '```' in line:
            item = line.split('`')[-1]
            if item:
                lines[i] = f'<pre><code class="{item}">'
            else:
                lines[i] = '</code></pre>'
        else:
            if i > 0:
                line = line.replace('<', '&lt;').replace('>', '&gt;')
                lines[i] = f'<br/>{line}'
    return ''.join(lines)



def clear_history(model_state, init_model_state):
    model_state = copy.deepcopy(init_model_state)
    return gr.update(value=[]), model_state


def init_chat_interface(chat_init_prompt):
    init_state = None
    init_tokens = tokenizer.tokenizer.encode(chat_init_prompt)
    tokens = init_tokens
    while len(tokens) > 0:
        out, init_state = model.forward(tokens[:256], init_state)
        tokens = tokens[256:]
    init_ctx = Context(init_state, init_state, [])
    init_model_state = init_state
    with gr.Blocks(css=css, analytics_enabled=False) as chat_interface:
        # chat_ctx=gr.State(init_ctx)
        model_state = gr.State(init_state)
        init_model_state = gr.State(init_state)
        last_model_state = gr.State(model_state)
        last_out = gr.State(out)
        chatbot = gr.Chatbot(elem_id='chatbot', show_label=False).style(height=450)
        message = gr.Textbox(
            show_label=False,
            placeholder="输入内容后按回车发送",
        ).style(container=False)
        # input_list = [message,chatbot,chat_ctx]
        # output_list = [message,chatbot,chat_ctx]
        with gr.Blocks(css=css, analytics_enabled=False):
            max_length = gr.Slider(0, 4096, value=2048, step=1.0, label="Maximum length", interactive=True)
            top_p = gr.Slider(0, 1, value=0.85, step=0.01, label="Top P", interactive=True)
            temperature = gr.Slider(0, 1, value=1, step=0.01, label="Temperature", interactive=True)

        message.submit(chat, [message, chatbot, model_state, max_length, top_p, temperature],
                       [message, chatbot, model_state, last_out, last_model_state])
        clear_history_btn = gr.Button('清空对话')
        regen_last_btn = gr.Button('重新生成上一条回答')
        clear_history_btn.click(clear_history, inputs=[model_state, init_model_state], outputs=[chatbot, model_state])
        regen_last_btn.click(regen_last, inputs=[chatbot, last_out, last_model_state, max_length, top_p, temperature], outputs=[chatbot, model_state])
    return chat_interface


def create_ui():
    chat_interface = init_chat_interface(init_prompt)
    tab_gen = types.SimpleNamespace()
    with gr.Blocks(css=css, analytics_enabled=False) as generate_interface:
        with gr.Row():
            with gr.Column():
                tab_gen.prompt = gr.Textbox(label="提示")
                tab_gen.generate_btn = gr.Button("生成")
            tab_gen.output = gr.Textbox(label="输出")

    interfaces = [
        (chat_interface, "Chat", "chat"),
        (generate_interface, "Generate", "generate")
    ]
    with gr.Blocks(css=css, analytics_enabled=False, title="ChatRWKV WebUI") as demo:
        gr.Markdown("""<h2><center>ChatRWKV WebUI</center></h2>""")
        with gr.Tabs(elem_id="tabs") as tabs:
            for interface, label, ifid in interfaces:
                with gr.TabItem(label, id=ifid, elem_id="tab_" + ifid):
                    interface.render()
        # chat_interface.render()

    return demo


# create_ui().queue().launch(server_name='0.0.0.0')
create_ui().queue().launch(server_name='0.0.0.0')

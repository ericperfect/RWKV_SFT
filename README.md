# RWKV_SFT

原项目在微调时构造数据集时是把所有数据拼接好，然后在数据集中随机取ctx_len个token作为一条数据，在单条数据文本过长时个人感觉会有些问题

基于此问题本项目简单重写了dataset，实现和常规sft训练输入相同（一条数据为一个qa对，计算loss时对q进行mask操作），此外增加了webui界面方面调试



## 基于Lora进行sft微调


### 训练数据样式

```json
{"instruction": "Q: 问题\n\nA: ", "input": "", "output": "答案\n\n"}
```
其中Q:和A: 和你推理时候的user、interface和bot保持一致即可。

```shell
cd RWKV-v4neo
sh train.sh
```
训练参数中my_qa_mask必须时1才能对问题进行mask, my_data_file为数据集地址，其余各参数基本没有变化，根据自己需求进行更改。

## webui

```shell
cd RWKV-v4neo
python webui.py
  --MODEL_NAME ''
  --MODEL_LORA ''
  --n_layer 32
  --n_embd 4096
  --ctx_len 4096
  --lora_r 16
  --lora_alpha 32
```
别的参数自行在RWKV-v4neo/webui.py中进行更改

### note
以上都是在单卡进行的实验，多卡没尝试过

## 致谢

本项目的开放过程中，获得了以下项目的帮助，在此表示感谢。

https://github.com/Blealtan/RWKV-LM-LoRA

https://github.com/BlinkDL/RWKV-LM

https://github.com/WuTianyi321/ChatRWKV-webui
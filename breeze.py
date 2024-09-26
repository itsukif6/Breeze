import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from fastapi import FastAPI, Header
import uvicorn
from pydantic import BaseModel
import time

model_name = "Breeze-7B"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    #from_tf=True
)

app = FastAPI()
class Message(BaseModel):
    title: str
    content: str

@app.post("/")
async def read_items(message: Message):
    s = time.time()
    print(message.title, message.content)

    prompt = message.content
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)

    output = model.generate(
    input_ids,
    # max_length=5,                 # 生成文本的最大長度
    max_new_tokens = 10,
    min_length=3,                  # 生成文本的最小長度
    do_sample=True,                 # 使用採樣而不是貪婪解碼
    num_beams=5,                    # 束搜索的束數
    temperature=0.2,                # 溫度參數，控制隨機性
    top_k=50,                       # 僅考慮前 k 個最可能的下一個詞
    top_p=0.95,                     # 核採樣參數
    no_repeat_ngram_size=2,         # 避免重複 n-gram
    num_return_sequences=3,         # 返回多個生成序列
    early_stopping=True,            # 當所有束都生成了 EOS 標記時停止
    #bad_words_ids=[[tokenizer.encode("不好", add_special_tokens=False)[0]]],  # 避免生成特定詞
    repetition_penalty=1.2,         # 重複懲罰因子
    length_penalty=1.0,             # 長度懲罰因子
    encoder_no_repeat_ngram_size=2, # 編碼器端避免重複 n-gram
    )
    # 解碼並輸出結果
    generated_text = tokenizer.decode(output[0], skip_special_tokens=False)
    e = time.time()
    print(generated_text)
    print(e - s)
    print(len(message.content))
    filterText = generated_text.split(" ")
    filterText.pop(0)
    filterText = ''.join(filterText)
    return {"message": filterText}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8091))
    uvicorn.run(app, host="0.0.0.0", port=port)
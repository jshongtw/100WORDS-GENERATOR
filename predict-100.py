from transformers import TextGenerationPipeline
from transformers import GPT2Tokenizer
from train import init_model, load_tokenizer
import string
import random
import datetime
from datetime import datetime

now = datetime.now()

tokenizer = load_tokenizer()
model = init_model(tokenizer)

text_generator = TextGenerationPipeline(model, tokenizer)

path = 'dataset/test/原始課程成果100字.txt'
pathout = 'dataset/test/ai100-chinese.txt'

f = open(path, 'r',encoding="utf-8")
fout = open(pathout, 'w',encoding="utf-8")

dtitle = '國文課程學習成果100字簡述- '+ now.strftime("%d/%m/%Y %H:%M:%S") + '\n\n'
fout.write(dtitle)

for k in range(1000):
  rand = random.randint(1,30)
  for r in range(rand):
    line = f.readline()

  input = line.strip('】\n')
  print(str(k).zfill(6),'原文: ',input)


  glist = text_generator((input[:7]), max_length=99, do_sample=True, top_k=5, repetition_penalty=1.3)
  generated = glist[0]["generated_text"]
  lastperiod = generated.rfind('。')
  print('last period 位置', lastperiod)
 
  sentence = str(k).zfill(6) + ': '+ generated[0:(lastperiod+1)] +  '\n'
  
  print(sentence)
  fout.write(sentence)


f.close()
fout.close()


#seed = ['魯迅是一個好作者','化學實驗真是好玩','我的物理考試很笨。','讀完一本書後，把自己的感想跟心得打出來','獨享，獨自享受，不願和其他人分享。']
#print(seed)
#for j in range(1):
#  input=seed[j]
#  print('段落',j)
#  for i in range(1):
#    glist = text_generator(input, max_length=95, do_sample=True, top_k=10, repetition_penalty=2.0)
#    gtext = glist[0]["generated_text"]
#    if input in gtext:
#      gtext = gtext.replace(input,'')
#    print(gtext)
#    input = gtext[-20:]


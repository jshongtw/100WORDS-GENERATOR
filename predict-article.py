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

path = 'dataset/test/ai100-chinese.txt'
pathout = 'dataset/test/ai-article-chinese.txt'

f = open(path, 'r',encoding="utf-8")
fout = open(pathout, 'w',encoding="utf-8")

dtitle = '國文課程學習成果100字簡述 '+ now.strftime("%d/%m/%Y %H:%M:%S") + '\n\n'
fout.write(dtitle)

authorlist = ["一桿「稱仔」", "賴和","一翦梅", "李清照","丁挽", "廖鴻基","十一月的白芒花", "楊牧","大山大河大海", "龍應台","大同與小康", "禮記","大德歌秋", "關漢卿","子路曾皙冉有公西華侍坐", "論語","小城連作", "鄭愁予","小寒", "向陽","工之僑為琴", "劉基","不繫之舟", "林泠","五十步笑百步", "孟子","五色令人目盲", "老子","公輸", "墨子","天下皆知美之為美", "老子","天才夢", "張愛玲","孔乙己", "魯迅","孔明借箭", "羅貫中","水調歌頭", "蘇軾","王道之始", "孟子","世說新語", "劉義慶","出師表", "諸葛亮","左忠毅公逸事", "方苞","正氣歌并序", "文天祥","玉山去來", "陳列","生年不滿百", "佚名","石壕吏", "杜甫","再別康橋", "徐志摩","江海所以能為百谷王", "老子","竹藪中", "芥川龍之介","老子選", "老子","壯遊", "楊牧","我的書齋", "鍾理和","折桂令九日", "張可久","牡丹亭．遊園", "湯顯祖","赤壁賦", "蘇軾","兒女", "朱自清","兒女", "豐子愷","典論論文", "曹丕","坤伶", "弦","夜雨寄北", "李商隱","始得西山宴遊記", "柳宗元","岳陽樓記", "范仲淹","庖丁解牛", "莊子","念奴嬌赤壁懷古", "蘇軾","明妃曲之一", "王安石","明湖居聽書", "劉鶚","河川證據", "簡媜","花和尚大鬧桃花村", "施耐庵","長干行", "李白","信言不美", "老子","剎那", "周夢蝶","客來小城", "鄭愁予","宣州謝朓樓餞別校書叔雲", "李白","恆久的滋味", "蔣勳","相見時難別亦難", "李商隱","看不見的珍藏", "褚威格","秋分", "向陽","范進中舉", "吳敬梓","郁離子選", "劉基","陌上桑", "佚名","飛魚季", "夏曼.藍波安","原君", "黃宗羲","唐傳奇選", "杜光庭","夏之絕句", "簡媜","師說", "韓愈","旅夜書懷", "杜甫","書憤", "陸游","桃花源記", "陶淵明","浪淘沙", "李煜","狼之獨步", "紀弦","破陣子為陳同甫賦壯詞以寄之", "辛棄疾","笑", "冰心","訓儉示康", "司馬光","送蔡培火蔣渭水陳逢源三君之京", "林幼春","馬鞍藤─憂傷西海岸之二", "吳晟","鬼頭刀", "廖鴻基","國葬", "白先勇","寄黃幾復", "黃庭堅","晚遊六橋待月記", "袁宏道","猛狗社鼠","韓非","第九味", "徐國能","聊齋志異選", "蒲松齡","雪夜訪戴", "劉義慶","勞山道士", "蒲松齡","散戲", "洪醒夫","棘刺刻猴", "韓非","琵琶行并序", "白居易","等你，在雨中", "余光中","絕妙好辭", "劉義慶","給我一個解釋", "張曉風","詠史其一", "左思","詠物篇選", "張曉風","詠絮之才", "劉義慶","陽關雪", "余秋雨","雁", "白萩","雁門關外，蕭捨命退遼兵", "金庸","項脊軒志", "歸有光","飲酒之五", "陶淵明","飲馬長城窟行", "佚名","馮諼客孟嘗君", "佚名","黃州快哉亭記", "蘇轍","黃鶴樓", "崔顥","廉恥", "顧炎武","愛", "張愛玲","愛之淚珠", "李黎","愛的辯證", "洛夫","溫州街到溫州街", "林文月","滅燭，憐光滿", "蔣勳","蜀賈三人", "劉基","詬食", "劉基","過秦論", "賈誼","漁父", "屈原","翡冷翠在下雨", "林文月","臺灣通史序", "連橫","與陳伯之書", "丘遲","蒙文課內蒙古篇", "席慕蓉","蒹葭", "佚名","蜘蛛之絲", "芥川龍之介","裨海紀遊選", "北投硫穴記","裨海紀遊選", "郁永河","趙人患鼠", "劉基","劉老老", "曹雪芹","墨子選", "墨子","慶東原", "白樸","蓼莪", "佚名","醉翁亭記", "歐陽脩","髯客傳", "杜光庭","魯智深大鬧桃花村", "施耐庵","戰士，乾杯！", "黃春明","戰國策", "佚名","諫太宗十思疏", "魏徵","諫逐客書", "李斯","錯誤", "鄭愁予","髻", "琦君","燭之武退秦師", "左丘明","韓非子選", "韓非","鴻門宴", "司馬遷","斷章", "卞之琳","離臺詩", "丘逢甲","藕神祠", "余光中","關於泰雅-出生禱詞", "瓦歷斯.諾幹","霧社", "瓦歷斯．諾幹","勸和論", "鄭用錫","勸學","荀子","釋迦果","沈光文","蘭亭集序","王羲之","驚螯","向陽","觀書有感之一","朱熹"]


prefixword = ['這次的','在經過','在這','其中，特別','透過','最讓','更令人','另外','藉由','我個人','我個人認為','這學期','我的','目前','不管','本文的目標在']
titleword = ['分析','我的想法','感動','我的探索']
first_titleword = ['前言','緣起','摘要','簡介','背景']
second_titleword = ['正文','本文']
last_titleword = ['心得','結論','反思','綜整心得']


 
seedtopic = random.choice(prefixword)

line = f.readline()
line = f.readline()

k=0
#  這邊是跳躍random行讀取原始100自簡述
while line:
    line = f.readline()
    #line = line[8:]
    # 跳過前面的編號

    original_line = line  
    print('--------------------------------------------------------------------------------------')
    print(original_line)
    fout.write(('---------'+ original_line+ '\n'))
     
    a = 0
    has_topic = 0

    for a in range(len(authorlist)):
        if (authorlist[a] in original_line):
            prefixword.append(authorlist[a]+"是")
            prefixword.append(authorlist[a]+"為")
            prefixword.append(authorlist[a]+"被")            
            has_topic = 1
            break
        else:
            a = a+1         
    
    if has_topic == 0:
        prefixword.append("這也讓我聯想到"+authorlist[random.randint(0,300)])
        prefixword.append("另一方面，"+authorlist[random.randint(0,300)])
        prefixword.append(authorlist[random.randint(0,300)]+"也是")

    seedtopic = random.choice(prefixword)
    feedin = seedtopic
#    print('feedin:', feedin)

    
#產出一段一段，段落數字random
    dofa = random.randint(2,4)
    for d in range(dofa):
        print("paragraphs:", str(dofa).zfill(2))
        fout.write(("paragraphs: "+ str(dofa).zfill(2) + '\n'))
        
        if d == 0:
            para_title= random.choice(first_titleword)
        elif d == 1:
            para_title = random.choice(second_titleword)   
        elif d == dofa-1:
            para_title = random.choice(last_titleword)
        else:
            para_title = ' '
            
            
        glist = text_generator(feedin, max_length=240, do_sample=True, top_k=10, repetition_penalty=1.1)
        glist_text = glist[0]["generated_text"]
        if ' ' in glist_text:
          glist_text = glist_text.replace(' ','')
        
        lastperiod = glist_text.rfind('。')
        glist_text = glist_text[0:(lastperiod+1)] +  '\n'
          
        output_sentence = str(k).zfill(6) + '第'+ str(d).zfill(2) + '段，標題: '+ para_title + '\n'+ glist_text + '\n' 
        
            
        print(output_sentence)
        fout.write((output_sentence+ '\n'))

        seedtopic = random.choice(prefixword)
        feedin = seedtopic

        
    k = k+1
    print()

    if k==2:
        break
 



f.close()
fout.close()





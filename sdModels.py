from diffusers import StableDiffusionPipeline, LMSDiscreteScheduler, EulerAncestralDiscreteScheduler
import torch
import urllib.request
import json
from keybert import KeyBERT
import keyconfig
import time

#번역이 필요한경우 
def Transword(inputtext):
        client_id = keyconfig.getData("client_id2") # 키가 저장된 파일은 비공개
        client_secret = keyconfig.getData("client_secret2") 
        encText = urllib.parse.quote(inputtext)
        data = "source=ko&target=en&text=" + encText
        url = "https://openapi.naver.com/v1/papago/n2mt"
        request = urllib.request.Request(url)
        request.add_header("X-Naver-Client-Id",client_id)
        request.add_header("X-Naver-Client-Secret",client_secret)
        response = urllib.request.urlopen(request, data=data.encode("utf-8"))
        rescode = response.getcode()
        if(rescode==200):
            response_body = response.read()
            result = response_body.decode('utf-8')
            d = json.loads(result)
            return(d['message']['result']['translatedText'])
            # print('응답타입: ', type(response_body))
        else:
            return("Error Code:" + rescode)

# 
def textkey(doc):
        """
        띄어쓰기 기준 (예: 1,2의 출력: 접근하지 남중국해)
        i : 최소추출
        j : 최대추출
        keyphrase_ngram_range(i,j)로 변경하여 사용
        """
        model = KeyBERT('distilbert-base-nli-mean-tokens')
        keywords = model.extract_keywords(doc,keyphrase_ngram_range=(1,1),stop_words=None)
        hesh = ""
        for i in keywords:
            hesh = hesh+" "+i[0]
        return hesh
        

#모델 실행 용
class sdModel:
    def __init__(self):
        self.doc = None
        self.pipe = None
        self.modelnum = None
        self.imageid = None

    def gettext(self, doc):
        self.doc = str(doc)

    def getimageid(self, num):
        self.imageid = str(num)

    def getmodel(self, modelnum):
        """
        modelnum == 1 -> 1.5v 기본
        modelnum == 2 -> 1.5v + 빙수님의 한글학습
        """
        if self.modelnum != int(modelnum):
            self.modelnum = int(modelnum)
            if self.modelnum == 1:
                self.pipe = torch.load("./models/model1.pt")
            elif self.modelnum == 2:
                self.pipe = torch.load("./models/model1.pt")
            elif self.modelnum == 3:
                pass
        else:
            pass

    def modelrun(self,savename):
        # doc, pipe가 로드되었는지 확인
        if self.doc == None:
            print("텍스트가 비었음")
        elif self.pipe == None:
            print("모델이 비었음")
        else:
            # 모델 2번인경우 예외처리
            # 모델에 만화처럼 그려주려면 promp 에 Comic style artwork 추가하도록 변경
            if self.modelnum==2:
                image = self.pipe(self.doc, num_inference_steps=25, generator=torch.Generator("cuda")).images[0]
                image.save(f'./static/imges/{str(savename)}.png', 'png')
            else:
                self.pipe = self.pipe.to("cuda")
                self.doc = Transword(self.doc)
                self.doc = textkey(self.doc)
                image = self.pipe(self.doc+"Comic style artwork").images[0]
                image.save(f'./static/imges/{str(savename)}.png', 'png')

if __name__ == "__main__":
    print("test 모드 실행되었습니다.")
    start = time.time()
    doc = """
    오늘은 비가 옵니다. 어떤 트럭을 운전하고있는 아져씨가 큰도로에 들어오게 됩니다. 
    그 아저씨는 도로를 달리다 어떤 여자가 도로 한 가운데에 서 있었습니다. 여다가 뒤를 돌자 선글라스를 끼고 서 있었습니다. 
    아저씨는 별거 아니라 생각하고 계속 달렸습니다. 그리고 또 그 여자가 그 선글라스를 끼고 똑같이 서 있습니다. 
    그 여자를 자세히 본 아저씨는 깜짝 놀랐습니다. 그 여자는 선글라스를 낀게 아니라 눈이 없었습니다. 깜짝 놀란 아저씨는 차로 달렸습니다. 
    그 여자는 그 트럭을 따라 옵니다. 다행이 아져씨가 아무 이상이 없었습니다. 
    집에 도착해 할걸 다 하고 거실에서 티비를 본고 있는데 뭔가 쎄 한 느낌이 들어서 옆을 봤는데 그 여자가 옆에 쇼파에 같이 앉아 있었습니다. 끝
    """
    sd_model = sdModel()
    sd_model.getmodel(1)
    sd_model.gettext(doc)
    print("선택된 모델은: ", sd_model.modelnum)
    sd_model.modelrun(99)
    print(sd_model.doc)

    end = time.time()
    print("소요 시간: ", end-start)





        


#papago and clova
import urllib.request
import keyconfig
import json
import requests
from keybert import KeyBERT

class transLengeaje():
         
    #파파고
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
            
    #클로바 <-- 정리필요
    def speech_rec(wav):
        clova_speech_client_id = keyconfig.getData("clova_speech_CLIENT_ID")
        clova_speech_client_secret = keyconfig.getData("clova_speech_CLIENT_SECRET")
        data = open(wav, "rb")
        Lang = "Kor" # Kor / Jpn / Chn / Eng
        URL = "https://naveropenapi.apigw.ntruss.com/recog/v1/stt?lang=" + Lang
        ID = clova_speech_client_id
        Secret = clova_speech_client_secret 
        headers = {
            "Content-Type": "application/octet-stream", # Fix
            "X-NCP-APIGW-API-KEY-ID": ID,
            "X-NCP-APIGW-API-KEY": Secret,
        }
        response = requests.post(URL,  data=data, headers=headers)
        rescode = response.status_code
        if(rescode == 200):
            result_text = response.text
            return (result_text)
        else:
            return ("Error : " + response.text)
        
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
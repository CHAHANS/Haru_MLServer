# **하루자국 프로젝트 _ AI서버**

## **프로젝트 개요**
---
1. 그림일기 [Web](https://github.com/TaeUkChu/VueDjango-Webpack1)에 필요한 AI 서버 제공(그림생성, 감성분석)
2. Web에 필요한 이미지 서버 제공

&nbsp;
## **사용법**
___
1. git-clone
2. keyconfig.py 생성
3. saveModels.py로 모델 생성 후 models 폴더로 이동
4. flask app 실행
5. 테스트

&nbsp;
## **기능**
---
1. **이미지 생성 및 감성분석 AI 실행**
    |요청 URL| 메서드 | 응답형식 | 설명 |
    |:---:|:----:|:---:|:---:|
    |/resultAPI|GET/POST|Json|{doc:요청문자,<br>imageId:이미지식별값, <br>modelId:실행 할 모델번호}|

2. **생성된 이미지 URL**
    |요청 URL| 메서드 | 응답형식 | 설명 |
    |:---:|:----:|:---:|:---:|
    |/imgview/imageId|GET|HTML||

3. **모델번호**
    |modelId|실행될 AI모델|
    |:---:|:---:|
    |1|stable-diffusion(diffusers 1.5)|
    |2|stable-diffusion(hugging face 빙수님 모델)|
    |3|stable-diffusion(fine-tuning)|
    |4|librosa (fine-tuning)|
    |5|kobert (fine-tuning)|

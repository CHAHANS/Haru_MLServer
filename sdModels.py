import torch
import time
import transLengeaje as L
        

#모델 실행 용
class sdModel(L.transLengeaje):
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
                self.pipe = torch.load("./models/model2.pt")
            elif self.modelnum == 3:
                self.pipe = torch.load("./models/model3.pt")
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
                self.doc = super().Transword(self.doc)
                self.doc = super().textkey(self.doc)
                if self.modelnum==3:
                    image = self.pipe(self.doc+" sketch").images[0]
                else:    
                    image = self.pipe(self.doc+" Comic style artwork").images[0]
                image.save(f'./static/imges/{str(savename)}.png', 'png')

if __name__ == "__main__":
    print("test 모드 실행되었습니다.")
    start = time.time()
    # doc = """
    # 오늘은 비가 옵니다. 어떤 트럭을 운전하고있는 아져씨가 큰도로에 들어오게 됩니다. 
    # 그 아저씨는 도로를 달리다 어떤 여자가 도로 한 가운데에 서 있었습니다. 여다가 뒤를 돌자 선글라스를 끼고 서 있었습니다. 
    # 아저씨는 별거 아니라 생각하고 계속 달렸습니다. 그리고 또 그 여자가 그 선글라스를 끼고 똑같이 서 있습니다. 
    # 그 여자를 자세히 본 아저씨는 깜짝 놀랐습니다. 그 여자는 선글라스를 낀게 아니라 눈이 없었습니다. 깜짝 놀란 아저씨는 차로 달렸습니다. 
    # 그 여자는 그 트럭을 따라 옵니다. 다행이 아져씨가 아무 이상이 없었습니다. 
    # 집에 도착해 할걸 다 하고 거실에서 티비를 본고 있는데 뭔가 쎄 한 느낌이 들어서 옆을 봤는데 그 여자가 옆에 쇼파에 같이 앉아 있었습니다. 끝
    # """

    doc = """
    과학기술정보통신부와 한국과학기술기획평가원이 전문가 논의를 거쳐 마련한 목록으로 2020년 4월 28일 공개됐다. 포스트 코로나는 코로나바이러스감염증-19 극복 이후 다가올 새로운 시대·상황을 이르는 말
    이며, 이 목록에는 포스트 코로나19 시대에 각광받을 유망기술 25개가 포함됐다.\r\n\r\n과학기술정보통신부에 따르면 전문가들은 4대 환경변화로 비대면·원격사회로의 전환, 바이오 시장의 새로운 도전과 기회, 자국
    중심주의 강화에 따른 글로벌 공급망 재편과 산업 스마트화 가속, 위험대응 일상화 및 회복력 중시 사회를 꼽았다. 이 4대 환경변화에 의해 큰 변화가 예상되는 사회·경제영역으로 헬스케어, 교육, 교통, 물류, 제조, 환경, 문화, 정보보안 등의 8개 영역을 선정했고 각 분야별로 5년 내에 현실화가 가능하면서 기술혁신성과 사회·경제적 파급효과가 큰 총 25개 유망기술을 나열했다.\r\n\r\n● 헬스케어: 디지털치료제, AI기반 실시 
    간 질병진단기술, 실시간 생체정보 측정·분석 기술, 감염병 확산 예측·조기경보기술, RNA바이러스 대항 백신기술 \r\n● 교육: 실감형 교육을 위한 가상·혼합현실 기술, AI·빅데이터 기반 맞춤형 학습 기술, 온라인수 
    업용 대용량 통신기술\r\n● 교통: 감염의심자 이송용 자율주행차, 개인맞춤형 라스트마일 모빌리티, 통합교통서비스(MaaS)\r\n● 물류: ICT기반 물류정보 통합플랫폼, 배송용 자율주행로봇, 유통물류센터 스마트화 기 
    술 \r\n● 제조: 디지털트윈, 인간증강기술, 협동로봇기술\r\n● 환경: 의료폐기물 수집·운반용 로봇, 인수공통감염병 통합관리기술 \r\n● 문화: 실감중계 서비스, 딥페이크 탐지기술, 드론기반의 GIS 구축 및 3D 영상 
    화 기술 \r\n● 정보보안: 화상회의 보안성 확보기술, 양자얽힘 기반의 화상보안통신기술, 동형암호 이용 동선추적시스템
    """
    sd_model = sdModel()
    sd_model.getmodel(3)
    sd_model.gettext(doc)

    print("선택된 모델은: ", sd_model.modelnum)
    sd_model.modelrun(103)
    print(sd_model.doc)

    end = time.time()
    print("소요 시간: ", end-start)





        


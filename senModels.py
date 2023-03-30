import numpy as np
import librosa # 음성 톤 분석 패키지
import pickle
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn

# 텍스트 감정분석
# from text_emotion_run import *

class ParallelModel(nn.Module):
    def __init__(self,num_emotions): # num_emotions 감정의 갯수
        super().__init__()

            # 1. conv block
        self.relu = nn.ReLU()
        self.conv1= nn.Conv2d(in_channels=1,
                   out_channels=16,
                   kernel_size=3,
                   stride=1,
                   padding=1
                  )
        self.bn1 = nn.BatchNorm2d(16)

        self.mp1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.do = nn.Dropout2d(p=0.3)
        # 2. conv block
        self.conv2= nn.Conv2d(in_channels=16,
                   out_channels=32,
                   kernel_size=3,
                   stride=1,
                   padding=1
                  )
        self.bn2 = nn.BatchNorm2d(32)
        self.mp2 = nn.MaxPool2d(kernel_size=4, stride=4)

        # 3. conv block
        self.conv3 = nn.Conv2d(in_channels=32,
                   out_channels=64,
                   kernel_size=3,
                   stride=1,
                   padding=1
                  )
        self.bn3 = nn.BatchNorm2d(64)

        # 4. conv block
        self.conv4= nn.Conv2d(in_channels=64,
                   out_channels=128,
                   kernel_size=3,
                   stride=1,
                   padding=1
                  )
        self.bn4= nn.BatchNorm2d(128)

        # Linear softmax layer
        self.out_linear = nn.Linear(512,num_emotions) # 선형으로 회귀
        self.out_softmax = nn.Softmax(dim=1)
        
    def forward(self,x):

        # transformer embedding
        out = self.relu(self.bn1(self.conv1(x)))

        out = self.do(self.mp1(out))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.do(self.mp2(out))
        out = self.relu(self.bn3(self.conv3(out)))
        out = self.do(self.mp2(out))
        out = self.relu(self.bn4(self.conv4(out)))
        out = self.do(self.mp2(out))


        conv_embedding = torch.flatten(out, start_dim=1)

        output_logits = self.out_linear(conv_embedding)
        output_softmax = self.out_softmax(output_logits)
        return output_logits, output_softmax  

# 아래는 예측할 때 음성파일을 전처리 해주는 함수들
# 음성 features를 반환해주는 함수 (1. windowing & sampling 과정을 통해 불러옴)
def load_audiofiles(file_name, sample_rate=48000): # sample rate : sampling시 1초에 몇 개의 sample을 추출하여 사용할지
    result=np.array([])
    audio_signal, sample_rate = librosa.load(file_name, duration=3, offset=0.5, sr=sample_rate)
    signal = np.zeros(int(sample_rate*3,))
    signal[:len(audio_signal)] = audio_signal
    return signal # 음성 features를 반환해준다.

# Melspectrogram 계산하는 함수
def Calculate_Melspectrogram(audio, sample_rate):
    mel_spec = librosa.feature.melspectrogram(y=audio,
                                              sr=sample_rate, # The sampling rate of y (number, default=22050)
                                              n_fft=1024, # The length of the Fourier transform window (number, default=2048)
                                              win_length = 512, # The length of the window function (number or None, default=n_fft)
                                              window='hamming', # The window function to use (string, tuple, or number, default='hann')
                                              hop_length = 256, # The number of samples between successive frames (number, default=512)
                                              n_mels=128, #  The number of mel frequency bands to use (number, default=128)
                                              fmax=sample_rate/2 # The highest frequency (in Hz) to use for the mel scale (number or None, default=sr/2)
                                             )
    
    # librosa.power_to_db : 소리 크기 -> log_scale로 변환
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max) # The reference power (number, default=1.0)
    return mel_spec_db

# 음성 톤 감정 예측하는 최종 함수
def emotion_voice(file_name):
    input_wav = str(file_name)
    signal = load_audiofiles(input_wav)
    mel_spectrogram = [Calculate_Melspectrogram(signal, sample_rate=48000)]
    mel_spectrogram = np.expand_dims(mel_spectrogram, 1)
    data = mel_spectrogram[0]
    data = torch.FloatTensor(data).to(device).unsqueeze(1)
    with torch.no_grad():
      softmax = model(data)[1]
    predictions = torch.argmax(softmax,dim=1).cpu().numpy()
    predicts = le.inverse_transform(predictions)
    return predicts

##main
# seed = 1
# np.random.seed(seed)
# torch.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ParallelModel(num_emotions=8).to(device)
model.load_state_dict(torch.load('/Users/User/Desktop/voice_emotion_recog_2/voice_emotion_recog_2/model_weights.pth'))

# 예측할 때 쓸 labelecoder fitting 시키기 위함
with open('/Users/User/Desktop/voice_emotion_recog_2/voice_emotion_recog_2/train_label.pkl', 'rb') as w:
    train_label = pickle.load(w)

le = LabelEncoder()
le.fit(train_label)

wav = "/Users/User/Desktop/voice_emotion_recog_2/voice_emotion_recog_2/testsample_1.wav"

# 음성 톤  감정분석
prediction = emotion_voice(wav)

if prediction == 'Angry':
  prediction = '분노'
if prediction == 'Anxious':
  prediction = '불안'
if prediction == 'Sad':
  prediction = '슬픔'
if prediction == 'Embarrassed':
  prediction = '당황'
if prediction == 'Hurt':
  prediction = '상처'
if prediction == 'Happy':
  prediction = '행복'
if prediction == 'Neutrality':
  prediction = '평온'


print('<음성 톤 감정분석 결과>')
print(f"당신의 목소리 톤에서 {prediction} 느껴집니다.")
print('')


# # 일기 입력하기
# dairy = speech_rec(wav)

# # 일기를 영어로 번역
# translate_result = Transword(dairy, 'ko', 'en')
# # 한 문장으로 요약
# summurized_diary = summarize_text(translate_result)

# #일기 감정 분석 결과
# emotion_result = predict(dairy)
# # 일기에서 감정을 영어로추출
# my_emotion = emotion_result[1]
# # 감정 결과를 보여줌
# emotion_print = emotion_result[0]
# # 요약과 감정을 받아 덕담 한 마디(영어 -> 한글 번역)
# word_for_me = give_word_for_me(summurized_diary, my_emotion)
# word_for_me_final = Transword(word_for_me, 'en', 'ko')

# print('')
# print('<일기 내용 감정분석 결과>')
# print(emotion_print)
# print('')
# print('-당신을 위한 한 마디-')
# print(word_for_me_final)
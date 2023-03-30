from diffusers import StableDiffusionPipeline, LMSDiscreteScheduler, EulerAncestralDiscreteScheduler
import torch

#기본 1.5 버전의 모델
model1= StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
torch.save(model1, "/Users/User/Desktop/Lab/project3/MLserver/models/model1.pt")
print('model1 저장')

# 빙수님의 커스텀 모델(한글 추가학습)
repo = "Bingsu/my-korean-stable-diffusion-v1-5"
euler_ancestral_scheduler = EulerAncestralDiscreteScheduler.from_config(repo, subfolder="scheduler")
model2 = StableDiffusionPipeline.from_pretrained(
repo, scheduler=euler_ancestral_scheduler, torch_dtype=torch.float16,
).to("cuda")
torch.save(model2, "/Users/User/Desktop/Lab/project3/MLserver/models/model2.pt")
print('model2 저장')

model_path = "/Users/User/Desktop/Lab/project3/MLserver/models/prototype_02.ckpt" # 체크포인트 경로
model_config = "runwayml/stable-diffusion-v1-5"  # 모델

model3 = StableDiffusionPipeline.from_pretrained(model_config, state_dict=torch.load(model_path, map_location="cpu"), torch_dtype=torch.float16)
torch.save(model3, "/Users/User/Desktop/Lab/project3/MLserver/models/model3.pt")
print('model3 저장')
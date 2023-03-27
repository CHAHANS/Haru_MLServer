from diffusers import StableDiffusionPipeline, LMSDiscreteScheduler, EulerAncestralDiscreteScheduler
import torch

#기본 1.5 버전의 모델
model1= StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)

# 빙수님의 커스텀 모델(한글 추가학습)
repo = "Bingsu/my-korean-stable-diffusion-v1-5"
euler_ancestral_scheduler = EulerAncestralDiscreteScheduler.from_config(repo, subfolder="scheduler")
model2 = StableDiffusionPipeline.from_pretrained(
repo, scheduler=euler_ancestral_scheduler, torch_dtype=torch.float16,
).to("cuda")

torch.save(model1, "./models/model1.pt")
torch.save(model2, "./models/model2.pt")
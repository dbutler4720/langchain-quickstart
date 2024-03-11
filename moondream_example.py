import torch
from transformers import pipeline, AutoModelForCausalLM, CodeGenTokenizerFast as Tokenizer
from PIL import Image

# This code works but it uses a LOT of memory
# I had to up the Docker Resources to Memory=>12GB, Swap=>3GB
# Took about 7.5 minutes to run the first time and 4.5 minutes on subsequent runs
device = torch.device("cpu")
dtype = torch.float32
model_id = "vikhyatk/moondream1"
print("device config complete")
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
print("model creation complete")
tokenizer = Tokenizer.from_pretrained(model_id)
print("tokenizer creationn complete")

image = Image.open('assets/demo-1.jpg')
print("image loaded")
enc_image = model.encode_image(image)
print("image encoded")
print(model.answer_question(enc_image, "Tell me about this picture.", tokenizer))
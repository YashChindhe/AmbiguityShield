import torch
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig

def load_vla_model(model_id="openvla/openvla-7b"):
    # Change compute_dtype to float16 for T4 compatibility
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16, 
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )
    
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    
    # Load model and force it to float16
    model = AutoModelForVision2Seq.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16  # Force the weights to match the bias type
    )
    return model, processor

def process_video_step(model, processor, img, instruction):
    # Ensure inputs are cast to float16 to match the model weights
    inputs = processor(instruction, img, return_tensors="pt").to("cuda", dtype=torch.float16)
    
    with torch.no_grad():
        outputs = model(**inputs)
        # Standard entropy calculation on the last token logits
        logits = outputs.logits[:, -1, :] 
        probs = torch.softmax(logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-9)).item()
    return entropy
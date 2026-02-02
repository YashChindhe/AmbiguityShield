import torch
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig

def load_vla_model(model_id="openvla/openvla-7b"):
    """Loads the OpenVLA model in 4-bit for T4 GPU compatibility."""
    # 1. Configure bitsandbytes for 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4"
    )
    
    # 2. Load Processor and Model
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    return model, processor

def calculate_entropy(logits):
    """Calculates Shannon Entropy of the action tokens."""
    # Convert logits to probabilities
    probs = torch.softmax(logits, dim=-1)
    # Calculate entropy: -sum(p * log(p))
    entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)
    return entropy.mean().item()

def process_video_step(model, processor, image, instruction):
    """Runs a single inference step and returns entropy."""
    inputs = processor(text=instruction, images=image, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        outputs = model(**inputs, output_scores=True, return_dict_in_generate=True)
        # Grab the logits for the first predicted action token
        logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
        
    entropy_score = calculate_entropy(logits)
    return entropy_score
from transformers import DonutProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch
import pdf2image
import os

# Load the pre-trained Donut model and processor
processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base")
model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base")

# Function to load an image or convert PDF to image
def load_image(file_path):
    if file_path.lower().endswith('.pdf'):
        # Convert PDF to a list of images
        images = pdf2image.convert_from_path(file_path)
        return images[0]  # Use the first page; modify if you need multiple pages
    else:
        # Load image file directly
        return Image.open(file_path).convert("RGB")

# Function to extract text using Donut
def extract_text_from_file(file_path):
    # Load the image
    image = load_image(file_path)

    # Prepare the image for the model
    pixel_values = processor(image, return_tensors="pt").pixel_values  # Shape: [1, 3, H, W]

    # Generate text (task prompt can be customized based on your needs)
    task_prompt = "<s_cord-v2>"  # Example prompt for general text extraction
    decoder_input_ids = processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids

    # Run the model
    with torch.no_grad():
        outputs = model.generate(
            pixel_values,
            decoder_input_ids=decoder_input_ids,
            max_length=512,  # Adjust based on expected output length
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
            use_cache=True,
            bad_words_ids=[[processor.tokenizer.unk_token_id]],
            return_dict_in_generate=True,
        )

    # Decode the generated sequence
    sequence = processor.batch_decode(outputs.sequences)[0]
    # Clean up the output (remove special tokens)
    extracted_text = sequence.replace(task_prompt, "").replace("</s>", "").strip()

    return extracted_text

# Example usage
if __name__ == "__main__":
    file_path = "data/img1.webp"  

    try:
        # Extract text
        text = extract_text_from_file(file_path)
        print("Extracted Text:")
        print(text)
    except Exception as e:
        print(f"An error occurred: {e}")
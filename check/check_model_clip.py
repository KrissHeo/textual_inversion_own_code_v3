from transformers import CLIPTokenizer, CLIPTextModel

tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
transformer = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
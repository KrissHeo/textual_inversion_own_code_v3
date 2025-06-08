from transformers import AutoTokenizer, AutoModel

model_name = "sentence-transformers/clip-ViT-B-32-multilingual-v1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

text = "노란색 컵이 테이블 위에 있다"
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)

word = "넨"

tokens = tokenizer.tokenize(word)

print(f"'{word}' → 토큰 수: {len(tokens)} | 토큰 내용: {tokens}")




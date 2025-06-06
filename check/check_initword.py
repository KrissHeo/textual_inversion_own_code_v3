# init_word로 설정하고자 하는 단어가 token 상 1-token인지 확인
from transformers import CLIPTokenizer

tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
word = input()
tokens = tokenizer.tokenize(word)
print(f"'{word}' → 토큰 수: {len(tokens)} | 토큰 내용: {tokens}")

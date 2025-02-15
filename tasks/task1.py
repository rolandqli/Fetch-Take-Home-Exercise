from sentence_transformer import SentenceTransformer 

# Test sentences
sentences = [
    "Well that's a bit far-fetched",
    "Go fetch boy",
    "Rewards, rewards, rewards"
]

# Model initialization
model = SentenceTransformer(tokenizer_name='bert-base-cased')

# Model inference
out = model(sentences)
print(out)
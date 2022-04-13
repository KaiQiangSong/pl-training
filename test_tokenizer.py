from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("facebook/bart-large")
idx = tok.encode("", return_tensors="pt")

print(idx)
print(idx.size())

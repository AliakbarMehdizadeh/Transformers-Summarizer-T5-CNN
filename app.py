from fastapi import FastAPI, Request
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

app = FastAPI()

# Loading model and tokenizer
#model_name = "path_to_your_finetuned_model"
model_name = "t5-base"  # or t5-small, t5-large depending on your needs
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Summarization endpoint
@app.post("/summarize/")
async def summarize_text(request: Request):
    body = await request.json()
    text = body['text']

    # Tokenize input text
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)

    # Generate summary
    summary_ids = model.generate(inputs, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return {"summary": summary}

from transformers import pipeline
import gradio as gr

model_checkpoint = "MuntasirHossain/distilbert-finetuned-ag-news"
model = pipeline("text-classification", model=model_checkpoint)


def classify(text):
    label = model(text)[0]["label"]
    return label

description = "This AI model is trained to classify texts/articles/sentences into four categories: World, Sports, Business and Science/Tech." 
title = "Classify Your Texts"
theme = "peach"
examples=[["Global Retail Giants Gear Up for Record-Breaking Holiday Sales Season Amidst Supply Chain Challenges and Rising Consumer Demand."]]

gr.Interface(fn=classify,
    inputs="textbox",
    outputs="text",
    title=title,
    theme = theme,
    description=description,
    examples=examples,
).launch()

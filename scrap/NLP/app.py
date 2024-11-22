import uvicorn
from fastapi import FastAPI, HTTPException, Query
from transformers import pipeline

app = FastAPI()

# Load the token classification model from Hugging Face
try:
    token_classifier = pipeline(
        "token-classification", model="Sarvina/fine_tuned_ner", aggregation_strategy="simple"
    )
except Exception as e:
    raise RuntimeError(f"Failed to load model from Hugging Face: {str(e)}")

# Load the sentiment analysis model from Hugging Face
try:
    sentiment_model = pipeline(
        "sentiment-analysis", model="Sarvina/fine_tuned_sentiment_analysis"  # Replace with your actual sentiment model
    )
except Exception as e:
    raise RuntimeError(f"Failed to load sentiment model from Hugging Face: {str(e)}")

@app.get('/')
def index():
    return {
        'message': "Hello, It is an API that handles NER and sentiment analysis tasks.",
        "For NER": "Input your text at the end of the URL: http://127.0.0.1:8000/ner/?data=",
        "For Sentiment Analysis": "Input your text at the end of the URL: http://127.0.0.1:8000/classify/?data="
    }

@app.get('/ner/')
def ner(data: str = Query(..., description="The text to perform NER on")):
    try:
        output = token_classifier(data)
        result = [{"word": i["word"], "entity": i["entity_group"]} for i in output]
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"NER error: {str(e)}")

@app.get('/classify/')
def classify(data: str = Query(..., description="The text to perform Sentiment Analysis on")):
    try:
        output = sentiment_model(data)
        result = [{"Sentiment": i["label"]} for i in output]
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sentiment analysis error: {str(e)}")

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)

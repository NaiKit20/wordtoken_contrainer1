from fastapi import FastAPI, HTTPException, Request
from pythainlp.corpus.common import thai_stopwords
from pythainlp import word_tokenize
import re
import emoji
import requests
from fastapi.middleware.cors import CORSMiddleware

# โหลดข้อมูลคำ stop word ภาษาไทยมา
thai_stopwords = list(thai_stopwords())
thai_stopwords

app = FastAPI()

@app.get("/api")
async def read_root():
    return {"message": "api start!!!"}

@app.post("/api/word_token")
async def word_token(data : Request):
    json = await data.json()
    text = json['text']

    token = emoji.demojize(text)
    token = "".join(u for u in token if u not in ("?", ".", ";", ":", "!", " ", "ๆ", "ฯ"))
    token = re.sub('[^A-Za-z0-9ก-๙]+', '', token)
    token = word_tokenize(token)
    token = " ".join(word for word in token)
    token = " ".join(word for word in token.split()
                     if word.lower not in thai_stopwords)
    
    # นำ('ข้อความ ที่ ได้') ส่งไปยัง container ที่ 2 เพื่อรับค่า predict
    url = "http://localhost:5000/api/predict"
    params = {"text": token}
    response = requests.post(url, json=params)
    
    return {"predict": response.json()['predict']}
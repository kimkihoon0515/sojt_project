from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import httpx
import asyncio


app = FastAPI()

origins = [
    "http://127.0.0.1",
    "http://127.0.0.1:5000"
]


URL = 'http://127.0.0.1:5000'

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*']
)

async def request(client):
    response = await client.get(URL)
    return response.text

async def task():
    async with httpx.AsyncClient() as client:
        task


@app.get("/")
async def root():
    return {"message": "Hello World"}

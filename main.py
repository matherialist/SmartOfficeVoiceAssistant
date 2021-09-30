import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from src.SmartOffice import SmartOffice

app = FastAPI()
so = SmartOffice()


class Data(BaseModel):
    text: str


@app.get('/recognise-command')
def recognise_command(data: Data):
    text = data.dict()['text']
    command, response = so.get_command(text)
    return {'command': command, 'response': response}


if __name__ == '__main__':
    uvicorn.run("main:app", host="0.0.0.0", port=5000, log_level="info")

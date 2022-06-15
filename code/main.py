from fastapi import FastAPI, File, UploadFile
import aiofiles
from pydantic import BaseModel
import NER_Model
import Negation_Detector

class Request(BaseModel):
    message: str
    language:str = "English"
    correct_spellings:bool = True
    negationStyle:str = "entities"

class Response(BaseModel):
    result:str

app = FastAPI()

@app.on_event("startup")
async def startup_event():
    print("****Application Started***")
    NER_Model.main()

@app.post("/extractInformation/")
async def create_item(req: Request):
    ner_result = NER_Model.predict(req.message, req.correct_spellings)
    result = Negation_Detector.predict(ner_result, req.negationStyle)
    return result

@app.post("/evaluate/")
async def create_item(req: Request):
    ner_result = NER_Model.predict(req.message, req.correct_spellings)
    result = Negation_Detector.predict(ner_result, req.negationStyle)
    return result

@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile):
    async with aiofiles.open("data/"+file.filename, 'wb') as out_file:
        content = await file.read()  # async read
        await out_file.write(content)  # async write

    return {"Result": file.filename + " Saved"}

@app.get("/")
async def root():
    return {"message": "Hello World"}
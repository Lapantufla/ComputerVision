
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import cv2 as cv
import numpy as np

from pipelines.predict_pipeline import predict_pipeline

app = FastAPI()

# Directorio donde se encuentran los templates HTML
#app.mount("/predict_html", StaticFiles(directory="htmls"), name="predict_html")
templates = Jinja2Templates(directory="./htmls")

@app.get("/", response_class=HTMLResponse)
async def upload_form(request: Request):
    
    # Renderiza la página HTML para cargar la foto
    return templates.TemplateResponse("predict_html.html", {"request": request})

@app.post("/make_prediction/")
async def predict_age(file: UploadFile = File(...)):
    contents = await file.read()
    
    # Convertir los bytes leídos en un numpy array
    nparr = np.fromstring(contents, np.uint8)
    
    # Decodificar la imagen para el uso con OpenCV
    img = cv.imdecode(nparr, cv.IMREAD_COLOR)

    edad_prediccion = predict_pipeline(img)

    if edad_prediccion == -1:
        print("entra en el error")
        return JSONResponse(content={'error': 'No se pudo cargar la imagen, intente con otra imagen.'})
    
    else:
        return JSONResponse(content={'prediction': f'La edad predicha es {edad_prediccion} años'})

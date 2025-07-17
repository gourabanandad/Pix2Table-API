from fastapi import FastAPI, File, UploadFile, HTTPException
import uvicorn
import os
import json
from fastapi.responses import HTMLResponse, JSONResponse
from bs4 import BeautifulSoup
from paddleocr import PPStructureV3
from datetime import datetime
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from fastapi.responses import HTMLResponse

templates = Jinja2Templates(directory="templates")
app = FastAPI()
pipeline = PPStructureV3()
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
def html_table_to_json(html: str, header_row_index: int = 0):
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table")

    if not table:
        return {"error": "No table found in HTML"}

    rows = table.find_all("tr")

    table_data = []
    num_cols = 0

    # Detect number of columns from first row with data
    for row in rows:
        cells = row.find_all(["td", "th"])
        if len(cells) > 0:
            num_cols = len(cells)
            break

    # Build headers (use first row or generate default headers)
    headers = []
    raw_header_cells = rows[header_row_index].find_all(["td", "th"])
    if len(raw_header_cells) == num_cols:
        headers = [cell.get_text(strip=True) or f"Column{i+1}" for i, cell in enumerate(raw_header_cells)]
        data_rows = rows[header_row_index + 1:]
    else:
        headers = [f"Column{i+1}" for i in range(num_cols)]
        data_rows = rows  # treat all rows as data

    # Parse rows
    for row in data_rows:
        cells = row.find_all(["td", "th"])
        row_data = [cell.get_text(strip=True) for cell in cells]
        if len(row_data) < num_cols:
            row_data += [""] * (num_cols - len(row_data))  # pad if cells missing
        elif len(row_data) > num_cols:
            row_data = row_data[:num_cols]  # truncate extras

        table_data.append(dict(zip(headers, row_data)))

    return table_data

async def process_image(image_path):
    if not os.path.exists(image_path):
        return {"error": "Image file does not exist"}

    output = pipeline.predict(input=image_path)
    if not output or 'table_res_list' not in output[0]:
        return {"error": "No table found in image"}

    html_content = output[0]['table_res_list'][0]['pred_html']
    json_data = html_table_to_json(html_content)

    return json_data
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
@app.post("/process-image")
async def process_image_endpoint(file: UploadFile = File(...)):
    if not allowed_file(file.filename):
        raise HTTPException(status_code=400, detail="Invalid file type. Only PNG, JPG, and JPEG are allowed.")
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(filepath, "wb") as buffer:
        buffer.write(await file.read())
    parsed_data = await process_image(filepath)
    os.remove(filepath)
    return JSONResponse({
        "status": "success",
        "message": "Image processed successfully",
        "data": parsed_data,
        "timestamp": datetime.now().isoformat()
    }
        
    )
    
    


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, port=8000)
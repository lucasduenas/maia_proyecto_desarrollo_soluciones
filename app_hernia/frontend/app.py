from flask import Flask, render_template, request, send_file
import requests
import base64
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

app = Flask(__name__)
API_URL = "http://localhost:8000/predict"

last_result = {}
last_image = None

@app.route("/", methods=["GET", "POST"])
def index():
    global last_result, last_image

    result = None
    image_preview = None

    if request.method == "POST":
        image = request.files["image"]
        image_bytes = image.read()
        last_image = image_bytes

        image_preview = base64.b64encode(image_bytes).decode("utf-8")

        response = requests.post(
            API_URL,
            files={"file": (image.filename, image_bytes, image.mimetype)}
        )

        if response.status_code == 200:
            result = response.json()
            last_result = result

    return render_template(
        "index.html",
        result=result,
        image_preview=image_preview
    )


@app.route("/export_pdf")
def export_pdf():
    buffer = BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    pdf.setFont("Helvetica-Bold", 18)
    pdf.drawString(50, height - 50, "Hernia Detection Report")

    pdf.setFont("Helvetica", 12)
    pdf.drawString(50, height - 100, f"Hernia Detected: {last_result.get('has_hernia')}")
    pdf.drawString(50, height - 130, f"Type: {last_result.get('type')}")
    pdf.drawString(50, height - 160, f"Confidence: {last_result.get('confidence') * 100}%")

    if last_image:
        image_stream = BytesIO(last_image)
        img = ImageReader(image_stream)

        pdf.drawImage(
            img,
            50,
            height - 450,
            width=300,
            preserveAspectRatio=True,
            mask='auto'
        )

    pdf.showPage()
    pdf.save()

    buffer.seek(0)
    return send_file(
        buffer,
        as_attachment=True,
        download_name="hernia_report.pdf",
        mimetype="application/pdf"
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000)
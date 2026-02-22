import os
import io
import uuid
import base64
from flask import Flask, render_template, request, send_file, session
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors

app = Flask(__name__)
app.secret_key = "hernia_secret_key"

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    image_preview = None

    if request.method == "POST":
        image = request.files["image"]

        # 1. Crear ID único por análisis
        analysis_id = str(uuid.uuid4())

        # 2. Leer imagen una sola vez
        image_bytes = image.read()

        # 3. Guardar imagen con ID único
        image_path = os.path.join(UPLOAD_FOLDER, f"{analysis_id}.png")
        with open(image_path, "wb") as f:
            f.write(image_bytes)

        # 4. Preview en base64
        image_preview = base64.b64encode(image_bytes).decode("utf-8")

        # 5. Inferencia (mock por ahora)
        result = {
            "analysis_id": analysis_id,
            "has_hernia": True,
            "confidence": 0.88
        }

        # 6. Guardar resultado en sesión
        session["last_result"] = result

    return render_template(
        "index.html",
        result=result,
        image_preview=image_preview
    )


@app.route("/export_pdf")
def export_pdf():
    result = session.get("last_result")

    if not result:
        return "No hay resultados para exportar", 400

    image_path = os.path.join(
        UPLOAD_FOLDER, f"{result['analysis_id']}.png"
    )

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )

    styles = getSampleStyleSheet()
    elements = []

    # ===== HEADER =====
    title = Paragraph(
        "<b>HernIA</b><br/><font size=11>"
        "Clasificación de hernias hiatales en imágenes radiológicas frontales</font>",
        ParagraphStyle(
            "title",
            fontSize=18,
            alignment=1,
            spaceAfter=20
        )
    )
    elements.append(title)

    # ===== IMAGEN =====
    elements.append(RLImage(image_path, width=320, height=320))
    elements.append(Spacer(1, 20))

    # ===== RESULTADOS =====
    status_color = colors.red if result["has_hernia"] else colors.green
    status_text = (
        "Hernia detectada" if result["has_hernia"]
        else "No se detectó hernia"
    )

    table = Table(
        [
            ["Resultado", status_text],
            ["Nivel de confianza", f"{int(result['confidence'] * 100)} %"]
        ],
        colWidths=[200, 200]
    )

    table.setStyle(TableStyle([
        ("TEXTCOLOR", (1, 0), (1, 0), status_color),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
        ("FONTSIZE", (0, 0), (-1, -1), 11),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
    ]))

    elements.append(table)
    elements.append(Spacer(1, 30))

    # ===== DISCLAIMER =====
    disclaimer = Paragraph(
        "<font size=9 color='#555555'>"
        "Esta aplicación tiene fines exclusivamente académicos y de investigación. "
        "Los resultados generados no deben ser utilizados para diagnóstico, tratamiento "
        "o toma de decisiones clínicas en situaciones médicas reales."
        "</font>",
        styles["Normal"]
    )

    elements.append(disclaimer)

    doc.build(elements)
    buffer.seek(0)

    return send_file(
        buffer,
        as_attachment=True,
        download_name="reporte_hernia.pdf",
        mimetype="application/pdf"
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000)

from flask import (
    Flask, render_template, request,
    redirect, url_for, flash
)

from database import get_db, init_db
from werkzeug.utils import secure_filename

from datetime import datetime
from pathlib import Path
import uuid

# ==========================================
# FLASK APPLICATION CONFIGURATION
# ==========================================

app = Flask(__name__)

app.secret_key = "smarttraffic_prototype_key"

BASE_DIR = Path(__file__).resolve().parent

UPLOAD_FOLDER = BASE_DIR / "static" / "uploads"
EVIDENCE_FOLDER = BASE_DIR / "static" / "evidence"

UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
EVIDENCE_FOLDER.mkdir(parents=True, exist_ok=True)

app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024

ALLOWED_VIDEO_EXTENSIONS = {
    "mp4", "avi", "mov", "mkv"
}

# Initialize database
init_db()


# ==========================================
# DASHBOARD
# ==========================================

@app.route("/")
def dashboard():

    conn = get_db()

    total_vehicles = conn.execute(
        "SELECT COUNT(*) FROM vehicles"
    ).fetchone()[0]

    total_violations = conn.execute(
        "SELECT COUNT(*) FROM violations"
    ).fetchone()[0]

    pending_challans = conn.execute(
        "SELECT COUNT(*) FROM violations WHERE status='Pending'"
    ).fetchone()[0]

    total_fines = conn.execute(
        "SELECT COALESCE(SUM(penalty), 0) FROM violations"
    ).fetchone()[0]

    recent_violations = conn.execute(
        "SELECT * FROM violations ORDER BY id DESC LIMIT 5"
    ).fetchall()

    conn.close()

    return render_template(
        "dashboard.html",
        total_vehicles=total_vehicles,
        total_violations=total_violations,
        pending_challans=pending_challans,
        total_fines=total_fines,
        recent_violations=recent_violations
    )


# ==========================================
# VIOLATIONS MANAGEMENT
# ==========================================

@app.route("/violations")
def violations():

    conn = get_db()

    records = conn.execute(
        "SELECT * FROM violations ORDER BY id DESC"
    ).fetchall()

    conn.close()

    return render_template(
        "violations.html",
        violations=records
    )


# ==========================================
# CREATE VIOLATION
# ==========================================

@app.route("/add-violation", methods=["POST"])
def add_violation():

    vehicle_number = request.form.get(
        "vehicle_number", ""
    ).strip().upper()

    violation_type = request.form.get(
        "violation_type", ""
    )

    location = request.form.get(
        "location", ""
    ).strip()

    penalties = {
        "No Helmet": 1000,
        "Red Light Violation": 1000,
        "Triple Riding": 1000
    }

    if not vehicle_number or not location:
        flash("Please enter all required details.", "error")
        return redirect(url_for("violations"))

    if violation_type not in penalties:
        flash("Invalid violation type.", "error")
        return redirect(url_for("violations"))

    conn = get_db()

    try:

        vehicle = conn.execute(
            "SELECT * FROM vehicles WHERE vehicle_number=?",
            (vehicle_number,)
        ).fetchone()

        if not vehicle:
            flash("Vehicle not found in demo database.", "error")
            return redirect(url_for("violations"))

        conn.execute("""
            INSERT INTO violations
            (
                vehicle_number,
                violation_type,
                location,
                violation_time,
                penalty,
                status
            )
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            vehicle_number,
            violation_type,
            location,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            penalties[violation_type],
            "Pending"
        ))

        conn.commit()

        flash(
            "Prototype e-challan generated successfully!",
            "success"
        )

        return redirect(url_for("challans"))

    finally:
        conn.close()


# ==========================================
# E-CHALLAN MANAGEMENT
# ==========================================

@app.route("/challans")
def challans():

    conn = get_db()

    try:

        records = conn.execute("""
            SELECT
                violations.*,
                vehicles.owner_name,
                vehicles.phone
            FROM violations
            LEFT JOIN vehicles
            ON violations.vehicle_number =
               vehicles.vehicle_number
            ORDER BY violations.id DESC
        """).fetchall()

    finally:
        conn.close()

    return render_template(
        "challan.html",
        challans=records
    )


# ==========================================
# CHALLAN DETAILS
# ==========================================

@app.route("/challan/<int:challan_id>")
def challan_detail(challan_id):

    conn = get_db()

    try:

        record = conn.execute("""
            SELECT
                violations.*,
                vehicles.owner_name,
                vehicles.phone,
                vehicles.vehicle_type,
                vehicles.insurance_status
            FROM violations
            LEFT JOIN vehicles
            ON violations.vehicle_number =
               vehicles.vehicle_number
            WHERE violations.id=?
        """, (challan_id,)).fetchone()

    finally:
        conn.close()

    if not record:
        return "Challan not found", 404

    return render_template(
        "challan_detail.html",
        challan=record
    )


# ==========================================
# AI VIDEO DETECTION
# ==========================================

@app.route("/ai-detection", methods=["GET", "POST"])
def ai_detection():

    detection_result = None
    error = None

    if request.method == "POST":

        video = request.files.get("video")

        if not video or not video.filename:

            error = "Please select a traffic video."

        else:

            filename = secure_filename(video.filename)

            if "." not in filename:

                error = "Invalid video filename."

            else:

                extension = filename.rsplit(".", 1)[-1].lower()

                if extension not in ALLOWED_VIDEO_EXTENSIONS:

                    error = "Unsupported video format."

                else:

                    unique_name = (
                        f"{uuid.uuid4().hex}.{extension}"
                    )

                    video_path = UPLOAD_FOLDER / unique_name

                    try:

                        # Save uploaded video
                        video.save(str(video_path))

                        # Import and execute AI engine
                        from ai_engine import process_video

                        detection_result = process_video(
                            video_path
                        )

                    except Exception as e:

                        print("AI Detection Error:", e)

                        error = (
                            "Video processing failed. "
                            "Check terminal for details."
                        )

                    finally:

                        # Delete uploaded video after processing
                        if video_path.exists():
                            video_path.unlink()

    return render_template(
        "ai_detection.html",
        result=detection_result,
        error=error
    )


# ==========================================
# MAXIMUM UPLOAD SIZE ERROR
# ==========================================

@app.errorhandler(413)
def file_too_large(error):

    return (
        "Video is too large. Maximum allowed size is 100 MB.",
        413
    )


# ==========================================
# APPLICATION ENTRY POINT
# ==========================================

if __name__ == "__main__":

    app.run(
        debug=True,
        host="127.0.0.1",
        port=5000
    )

# SignPay — Behavioral Signature Authentication for UPI
# Author: Manish Kumar K M
# License: MIT
# Description: Flask backend for on-device signature matching
"""
SignPay Flask Backend
=====================
API Endpoints:
  POST /enroll   → save user's signature (.sgpx)
  POST /verify   → compare drawn signature vs stored
  POST /reset    → delete old .sgpx, save new one
  GET  /status   → check if user has enrolled signature
  GET  /sgpx     → get .sgpx file summary (for demo display)
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from signature_engine import (
    png_to_binary_grid,
    save_sgpx,
    load_sgpx,
    delete_sgpx,
    compare_signatures,
    has_enough_ink,
    get_sgpx_summary
)

app = Flask(__name__)
CORS(app)  # Allow frontend (HTML file) to call this API

# Default user for demo
DEFAULT_USER = "demo_user"


# ─────────────────────────────────────────
# POST /enroll
# Body: { png: "data:image/png;base64,...", timing: [...] }
# Saves signature as .sgpx file
# ─────────────────────────────────────────
@app.route('/enroll', methods=['POST'])
def enroll():
    try:
        data = request.get_json()
        png_b64 = data.get('png')
        timing = data.get('timing', [])
        user_id = data.get('user_id', DEFAULT_USER)

        if not png_b64:
            return jsonify({"success": False, "error": "No image data"}), 400

        # Convert canvas PNG → binary grid
        grid = png_to_binary_grid(png_b64)
        ink_count = sum(grid)

        # Check enough ink
        if not has_enough_ink(grid):
            return jsonify({
                "success": False,
                "error": "Signature too short — draw more"
            }), 400

        # Save to .sgpx file
        filepath = save_sgpx(user_id, grid, timing, ink_count)

        # Return summary
        summary = get_sgpx_summary(user_id)

        return jsonify({
            "success": True,
            "message": f"Signature enrolled for {user_id}",
            "ink_pixels": ink_count,
            "timing_points": len(timing),
            "file": filepath,
            "sgpx_summary": summary
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ─────────────────────────────────────────
# POST /verify
# Body: { png: "...", timing: [...] }
# Compares against stored .sgpx
# ─────────────────────────────────────────
@app.route('/verify', methods=['POST'])
def verify():
    try:
        data = request.get_json()
        png_b64 = data.get('png')
        timing = data.get('timing', [])
        user_id = data.get('user_id', DEFAULT_USER)

        if not png_b64:
            return jsonify({"success": False, "error": "No image data"}), 400

        # Load enrolled signature
        enrolled = load_sgpx(user_id)
        if not enrolled:
            return jsonify({
                "success": False,
                "error": "No signature enrolled. Please set up first."
            }), 404

        # Convert test drawing → grid
        test_grid = png_to_binary_grid(png_b64)

        if not has_enough_ink(test_grid):
            return jsonify({
                "success": False,
                "matched": False,
                "error": "Signature too short — draw more",
                "score": 0
            }), 400

        # Compare
        result = compare_signatures(enrolled, test_grid, timing)

        # Log to console (so you can see what's happening)
        print(f"\n[VERIFY] User: {user_id}")
        print(f"Shape score:  {result['shape_score']}%")
        print(f"  Timing score: {result['timing_score']}%")
        print(f"  Combined:     {result['score']}%")
        print(f"  Result:       {result['verdict']}")

        return jsonify({
            "success": True,
            **result
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ─────────────────────────────────────────
# POST /reset
# Body: { png: "...", timing: [...] }
# Deletes old .sgpx and saves new one
# ─────────────────────────────────────────
@app.route('/reset', methods=['POST'])
def reset():
    try:
        data = request.get_json()
        png_b64 = data.get('png')
        timing = data.get('timing', [])
        user_id = data.get('user_id', DEFAULT_USER)

        if not png_b64:
            return jsonify({"success": False, "error": "No image data"}), 400

        grid = png_to_binary_grid(png_b64)
        ink_count = sum(grid)

        if not has_enough_ink(grid):
            return jsonify({
                "success": False,
                "error": "New signature too short"
            }), 400

        # Delete old
        deleted = delete_sgpx(user_id)

        # Save new
        filepath = save_sgpx(user_id, grid, timing, ink_count)
        summary = get_sgpx_summary(user_id)

        return jsonify({
            "success": True,
            "message": "Signature reset successfully",
            "old_deleted": deleted,
            "new_file": filepath,
            "sgpx_summary": summary
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ─────────────────────────────────────────
# GET /status?user_id=demo_user
# Check if user has enrolled
# ─────────────────────────────────────────
@app.route('/status', methods=['GET'])
def status():
    user_id = request.args.get('user_id', DEFAULT_USER)
    enrolled = load_sgpx(user_id)
    return jsonify({
        "enrolled": enrolled is not None,
        "user_id": user_id,
        "summary": get_sgpx_summary(user_id) if enrolled else None
    })


# ─────────────────────────────────────────
# GET /sgpx?user_id=demo_user
# Get .sgpx file summary for display
# ─────────────────────────────────────────
@app.route('/sgpx', methods=['GET'])
def sgpx_info():
    user_id = request.args.get('user_id', DEFAULT_USER)
    summary = get_sgpx_summary(user_id)
    enrolled = load_sgpx(user_id)
    return jsonify({
        "summary": summary,
        "has_signature": enrolled is not None
    })


if __name__ == '__main__':
    print("=" * 50)
    print("  SignPay Backend Running")
    print("  http://localhost:5000")
    print("=" * 50)
    print("Endpoints:")
    print("  POST /enroll  — save signature")
    print("  POST /verify  — match signature")
    print("  POST /reset   — change signature")
    print("  GET  /status  — check enrollment")
    print("  GET  /sgpx    — view .sgpx info")
    print("=" * 50)
    app.run(debug=True, port=5000)

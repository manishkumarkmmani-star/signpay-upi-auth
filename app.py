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

CORS(app)

DEFAULT_USER = "demo_user"


# ============================================================
# ENROLL
# ============================================================

@app.route("/enroll", methods=["POST"])
def enroll():

    try:

        data = request.get_json() or {}

        png_b64 = data.get("png")
        timing = data.get("timing", [])
        user_id = data.get(
            "user_id",
            DEFAULT_USER
        )

        if not png_b64:

            return jsonify({
                "success": False,
                "error": "No image data"
            }), 400

        grid = png_to_binary_grid(
            png_b64
        )

        if not has_enough_ink(grid):

            return jsonify({
                "success": False,
                "error":
                    "Signature too short — draw more"
            }), 400

        filepath = save_sgpx(
            user_id,
            grid,
            timing,
            sum(grid)
        )

        return jsonify({
            "success": True,
            "message":
                f"Signature enrolled for {user_id}",
            "file": filepath,
            "sgpx_summary":
                get_sgpx_summary(user_id)
        })

    except Exception as e:

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# ============================================================
# VERIFY
# ============================================================

@app.route("/verify", methods=["POST"])
def verify():

    try:

        data = request.get_json() or {}

        png_b64 = data.get("png")
        timing = data.get("timing", [])

        user_id = data.get(
            "user_id",
            DEFAULT_USER
        )

        if not png_b64:

            return jsonify({
                "success": False,
                "error": "No image data"
            }), 400

        enrolled = load_sgpx(
            user_id
        )

        if not enrolled:

            return jsonify({
                "success": False,
                "error":
                    "No signature enrolled"
            }), 404

        test_grid = png_to_binary_grid(
            png_b64
        )

        if not has_enough_ink(test_grid):

            return jsonify({
                "success": True,
                "matched": False,
                "score": 0,
                "error":
                    "Signature too short — draw more"
            })

        result = compare_signatures(
            enrolled,
            test_grid,
            timing
        )

        print("\n========== SIGNPAY VERIFY ==========")

        print(
            "Shape:",
            result["shape_score"]
        )

        print(
            "  IoU:",
            result["shape"]["iou"]
        )

        print(
            "  OpenCV:",
            result["shape"]["opencv"]
        )

        print(
            "  Hu:",
            result["shape"]["hu"]
        )

        print(
            "Behavior:",
            result["behavior_score"]
        )

        print(
            "  Trajectory:",
            result["behavior"]["trajectory"]
        )

        print(
            "  Velocity:",
            result["behavior"]["velocity"]
        )

        print(
            "  Direction:",
            result["behavior"]["direction"]
        )

        print(
            "  Duration:",
            result["behavior"]["duration"]
        )

        print(
            "  Stroke:",
            result["behavior"]["stroke"]
        )

        print(
            "FINAL:",
            result["score"]
        )

        print(
            "THRESHOLD:",
            result["threshold"]
        )

        print(
            "RESULT:",
            result["verdict"]
        )

        print("====================================")

        return jsonify({
            "success": True,
            **result
        })

    except Exception as e:

        print(
            "[SignPay ERROR]",
            repr(e)
        )

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# ============================================================
# RESET
# ============================================================

@app.route("/reset", methods=["POST"])
def reset():

    try:

        data = request.get_json() or {}

        png_b64 = data.get("png")
        timing = data.get("timing", [])

        user_id = data.get(
            "user_id",
            DEFAULT_USER
        )

        if not png_b64:

            return jsonify({
                "success": False,
                "error": "No image data"
            }), 400

        grid = png_to_binary_grid(
            png_b64
        )

        if not has_enough_ink(grid):

            return jsonify({
                "success": False,
                "error":
                    "New signature too short"
            }), 400

        deleted = delete_sgpx(
            user_id
        )

        filepath = save_sgpx(
            user_id,
            grid,
            timing,
            sum(grid)
        )

        return jsonify({
            "success": True,
            "old_deleted": deleted,
            "new_file": filepath,
            "sgpx_summary":
                get_sgpx_summary(user_id)
        })

    except Exception as e:

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# ============================================================
# STATUS
# ============================================================

@app.route("/status", methods=["GET"])
def status():

    user_id = request.args.get(
        "user_id",
        DEFAULT_USER
    )

    enrolled = load_sgpx(
        user_id
    )

    return jsonify({
        "enrolled":
            enrolled is not None,
        "user_id":
            user_id,
        "summary":
            get_sgpx_summary(user_id)
            if enrolled
            else None
    })


# ============================================================
# SGPX INFO
# ============================================================

@app.route("/sgpx", methods=["GET"])
def sgpx_info():

    user_id = request.args.get(
        "user_id",
        DEFAULT_USER
    )

    enrolled = load_sgpx(
        user_id
    )

    return jsonify({
        "summary":
            get_sgpx_summary(user_id),
        "has_signature":
            enrolled is not None
    })


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":
    print("=" * 50)
    print(" SignPay Backend Running")
    print(" http://127.0.0.1:5000")
    print("=" * 50)

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )
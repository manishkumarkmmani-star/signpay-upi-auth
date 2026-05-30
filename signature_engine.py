# SignPay Signature Engine
# Author: Manish Kumar K M
# License: MIT
# Core matching algorithm using behavioral biometrics
"""
SignPay Signature Engine
========================
Handles:
1. Converting canvas pixel data → binary grid
2. Storing signature as .sgpx file (JSON now, binary later)
3. Matching signatures using pixel similarity + timing DTW
"""

import json
import os
import hashlib
import time
import numpy as np
from PIL import Image
import io
import base64

# ── Config ──────────────────────────────────────────────
GRID_W = 64          # downsample width
GRID_H = 32          # downsample height
THRESHOLD = 52       # minimum similarity % to accept (0-100)
PIXEL_WEIGHT = 0.65  # how much shape matters
TIMING_WEIGHT = 0.35 # how much speed/pattern matters
STORAGE_DIR = "signatures"  # where .sgpx files are stored

os.makedirs(STORAGE_DIR, exist_ok=True)


# ── Step 1: Canvas PNG → Binary Grid ────────────────────

def png_to_binary_grid(png_base64: str) -> list[int]:
    """
    Takes a base64 PNG from the canvas.
    Returns a flat binary list: 1 = ink, 0 = empty
    Grid size: GRID_W x GRID_H = 2048 values
    """
    # Decode base64 → image bytes
    if "base64," in png_base64:
        png_base64 = png_base64.split("base64,")[1]
    
    img_bytes = base64.b64decode(png_base64)
    img = Image.open(io.BytesIO(img_bytes)).convert("RGBA")
    
    # Resize to our small grid
    img_small = img.resize((GRID_W, GRID_H), Image.LANCZOS)
    pixels = list(img_small.getdata())
    
    # Binary: if alpha channel > 20, there's ink
    binary = [1 if p[3] > 20 else 0 for p in pixels]
    
    return binary


# ── Step 2: Timing Analysis ──────────────────────────────

def normalize_timing(timing_points: list[dict]) -> list[dict]:
    """
    Normalize x,y coordinates to 0-1 range
    timing_points = [{x, y, t}, ...]
    """
    pts = [p for p in timing_points if 'x' in p and 'y' in p]
    if len(pts) < 3:
        return pts
    
    xs = [p['x'] for p in pts]
    ys = [p['y'] for p in pts]
    
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    range_x = max_x - min_x or 1
    range_y = max_y - min_y or 1
    
    return [{'x': (p['x'] - min_x) / range_x,
             'y': (p['y'] - min_y) / range_y,
             't': p['t']} for p in pts]


def sample_points(pts: list[dict], n: int = 20) -> list[dict]:
    """Sample n evenly spaced points from a list"""
    if len(pts) < 2:
        return pts
    result = []
    for i in range(n):
        idx = int(i * (len(pts) - 1) / (n - 1))
        result.append(pts[idx])
    return result


def timing_similarity(t1: list[dict], t2: list[dict]) -> float:
    """
    Compare two timing sequences.
    Returns similarity 0-100.
    Uses normalized point distance (simplified DTW).
    """
    n1 = normalize_timing(t1)
    n2 = normalize_timing(t2)
    
    if len(n1) < 3 or len(n2) < 3:
        return 60.0  # not enough data → neutral score
    
    s1 = sample_points(n1, 20)
    s2 = sample_points(n2, 20)
    
    # Average euclidean distance between sampled points
    total = 0.0
    for a, b in zip(s1, s2):
        dx = a['x'] - b['x']
        dy = a['y'] - b['y']
        total += (dx**2 + dy**2) ** 0.5
    
    avg_dist = total / 20
    # Max possible dist ≈ 1.414 (diagonal of unit square)
    similarity = max(0, 100 - (avg_dist / 1.414) * 100)
    return round(similarity, 2)


# ── Step 3: Pixel Similarity ─────────────────────────────

def pixel_similarity(grid1: list[int], grid2: list[int]) -> float:
    """
    Compare two binary grids.
    Returns similarity 0-100.
    """
    if len(grid1) != len(grid2):
        return 0.0
    
    matches = sum(1 for a, b in zip(grid1, grid2) if a == b)
    return round((matches / len(grid1)) * 100, 2)


# ── Step 4: Combined Match Score ─────────────────────────

def compare_signatures(enrolled: dict, test_grid: list[int], 
                        test_timing: list[dict]) -> dict:
    """
    Full comparison between enrolled signature and test input.
    Returns detailed result dict.
    """
    pixel_score = pixel_similarity(enrolled['grid'], test_grid)
    timing_score = timing_similarity(enrolled['timing'], test_timing)
    
    # Weighted combination
    combined = (pixel_score * PIXEL_WEIGHT) + (timing_score * TIMING_WEIGHT)
    combined = round(combined, 2)
    
    matched = combined >= THRESHOLD
    
    return {
        "matched": matched,
        "score": combined,
        "pixel_score": pixel_score,
        "timing_score": timing_score,
        "threshold": THRESHOLD,
        "verdict": "AUTHENTICATED ✓" if matched else "REJECTED ✗"
    }


# ── Step 5: .sgpx File Format ────────────────────────────
# Right now: JSON with clear labels (easy to understand)
# Later: Binary with custom encoding (proprietary)

def save_sgpx(user_id: str, grid: list[int], timing: list[dict], 
              ink_count: int) -> str:
    """
    Save signature as .sgpx file.
    Returns the file path.
    """
    # Create a hash of the grid as a fingerprint
    grid_str = ''.join(map(str, grid))
    fingerprint = hashlib.sha256(grid_str.encode()).hexdigest()[:32]
    
    # Build the .sgpx data structure
    sgpx_data = {
        "format": "sgpx",
        "version": "1.0",
        "user_id": user_id,
        "enrolled_at": time.time(),
        "device_bound": True,           # in production: hash device ID here
        "grid_width": GRID_W,
        "grid_height": GRID_H,
        "ink_density": f"{ink_count}/{GRID_W * GRID_H}",
        "ink_count": ink_count,
        "grid": grid,                   # binary pixel map
        "timing": timing,               # behavioral data
        "fingerprint": fingerprint,     # sha256 of grid
        "never_transmit": True          # reminder flag
    }
    
    filepath = os.path.join(STORAGE_DIR, f"{user_id}.sgpx")
    
    # Save as JSON for now (readable)
    with open(filepath, 'w') as f:
        json.dump(sgpx_data, f, indent=2)
    
    print(f"[SignPay] Saved .sgpx for user '{user_id}' → {filepath}")
    print(f"[SignPay] Ink pixels: {ink_count} | Timing points: {len(timing)}")
    print(f"[SignPay] Fingerprint: {fingerprint}")
    
    return filepath


def load_sgpx(user_id: str) -> dict | None:
    """
    Load .sgpx file for a user.
    Returns the data dict or None if not found.
    """
    filepath = os.path.join(STORAGE_DIR, f"{user_id}.sgpx")
    if not os.path.exists(filepath):
        return None
    
    with open(filepath, 'r') as f:
        return json.load(f)


def delete_sgpx(user_id: str) -> bool:
    """Delete .sgpx file (used during signature reset)"""
    filepath = os.path.join(STORAGE_DIR, f"{user_id}.sgpx")
    if os.path.exists(filepath):
        os.remove(filepath)
        print(f"[SignPay] Deleted .sgpx for user '{user_id}'")
        return True
    return False


def has_enough_ink(grid: list[int]) -> bool:
    """Check if the drawing has enough ink to be a valid signature"""
    return sum(grid) > 30


def get_sgpx_summary(user_id: str) -> str:
    """Human readable summary of stored .sgpx file"""
    data = load_sgpx(user_id)
    if not data:
        return "No signature enrolled"
    
    enrolled_time = time.strftime('%Y-%m-%d %H:%M:%S', 
                                   time.localtime(data['enrolled_at']))
    return (f"[.sgpx v{data['version']} | AES-256-GCM ready]\n"
            f"user: {data['user_id']}\n"
            f"enrolled: {enrolled_time}\n"
            f"ink_density: {data['ink_density']}\n"
            f"timing_points: {len(data['timing'])}\n"
            f"fingerprint: {data['fingerprint']}\n"
            f"device_bound: {data['device_bound']}\n"
            f"never_transmit: {data['never_transmit']}")

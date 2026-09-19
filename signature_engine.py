# SignPay Signature Engine
# Behavioral Signature Authentication for UPI

import json
import os
import hashlib
import time
import base64
import io
import math

import cv2
import numpy as np
from PIL import Image


# ============================================================
# CONFIGURATION
# ============================================================

GRID_W = 64
GRID_H = 32

# Final decision threshold.
# DO NOT treat this as "accuracy".
# We will calibrate it later using FAR/FRR data.
THRESHOLD = 75.0

# Fusion weights.
# Visual shape is important, but behavior must contribute strongly.
SHAPE_WEIGHT = 0.6
BEHAVIOR_WEIGHT = 0.4

STORAGE_DIR = "signatures"

os.makedirs(STORAGE_DIR, exist_ok=True)


# ============================================================
# 1. IMAGE PREPROCESSING
# ============================================================

def png_to_binary_grid(png_base64: str) -> list[int]:
    """
    Convert canvas PNG into a normalized 64x32 binary signature grid.
    """

    if "base64," in png_base64:
        png_base64 = png_base64.split("base64,", 1)[1]

    img_bytes = base64.b64decode(png_base64)
    img = Image.open(io.BytesIO(img_bytes)).convert("RGBA")

    # Downsample first.
    img_small = img.resize((GRID_W, GRID_H), Image.LANCZOS)

    pixels = list(img_small.getdata())

    binary = [
        1 if pixel[3] > 20 else 0
        for pixel in pixels
    ]

    binary = crop_to_ink(binary)
    binary = center_align(binary)

    return binary


def crop_to_ink(
    grid: list[int],
    width: int = GRID_W,
    height: int = GRID_H
) -> list[int]:
    """
    Crop signature ink while PRESERVING aspect ratio.

    Important:
    - Removes large empty borders.
    - Does NOT stretch the signature independently
      in X and Y.
    - Fits the signature inside the 64x32 grid.
    - Adds a small safety margin.
    """

    rows = [
        grid[i * width:(i + 1) * width]
        for i in range(height)
    ]

    # --------------------------------------------------------
    # Find ink bounding box
    # --------------------------------------------------------

    ink_rows = [
        i for i, row in enumerate(rows)
        if any(v == 1 for v in row)
    ]

    ink_cols = [
        j for j in range(width)
        if any(rows[i][j] == 1 for i in range(height))
    ]

    # Empty signature
    if not ink_rows or not ink_cols:
        return grid

    top = ink_rows[0]
    bottom = ink_rows[-1]
    left = ink_cols[0]
    right = ink_cols[-1]

    crop_h = bottom - top + 1
    crop_w = right - left + 1

    # --------------------------------------------------------
    # Add small padding around the signature
    # --------------------------------------------------------

    PADDING = 1

    top = max(0, top - PADDING)
    bottom = min(height - 1, bottom + PADDING)
    left = max(0, left - PADDING)
    right = min(width - 1, right + PADDING)

    crop_h = bottom - top + 1
    crop_w = right - left + 1

    cropped = np.array(
        [
            rows[i][j]
            for i in range(top, bottom + 1)
            for j in range(left, right + 1)
        ],
        dtype=np.uint8
    ).reshape(crop_h, crop_w)

    # --------------------------------------------------------
    # Preserve aspect ratio
    # --------------------------------------------------------

    margin_x = 4
    margin_y = 3

    available_w = max(1, width - 2 * margin_x)
    available_h = max(1, height - 2 * margin_y)

    scale = min(
        available_w / crop_w,
        available_h / crop_h
    )

    new_w = max(
        1,
        min(
            available_w,
            int(round(crop_w * scale))
        )
    )

    new_h = max(
        1,
        min(
            available_h,
            int(round(crop_h * scale))
        )
    )

    resized = cv2.resize(
        cropped,
        (new_w, new_h),
        interpolation=cv2.INTER_NEAREST
    )

    # --------------------------------------------------------
    # Put resized signature into empty 64x32 canvas
    # --------------------------------------------------------

    result = np.zeros(
        (height, width),
        dtype=np.uint8
    )

    start_x = (width - new_w) // 2
    start_y = (height - new_h) // 2

    result[
        start_y:start_y + new_h,
        start_x:start_x + new_w
    ] = resized

    return result.flatten().tolist()


def center_align(
    grid: list[int],
    width: int = GRID_W,
    height: int = GRID_H
) -> list[int]:
    """
    Align the signature using its BOUNDING-BOX CENTER.

    This is more stable than center-of-mass alignment because
    different handwriting strokes can shift the ink mass even
    when the actual signature geometry is similar.
    """

    rows = [
        grid[i * width:(i + 1) * width]
        for i in range(height)
    ]

    # --------------------------------------------------------
    # Find bounding box
    # --------------------------------------------------------

    ink_rows = [
        i for i, row in enumerate(rows)
        if any(v == 1 for v in row)
    ]

    ink_cols = [
        j for j in range(width)
        if any(rows[i][j] == 1 for i in range(height))
    ]

    if not ink_rows or not ink_cols:
        return grid

    top = ink_rows[0]
    bottom = ink_rows[-1]
    left = ink_cols[0]
    right = ink_cols[-1]

    # --------------------------------------------------------
    # Bounding-box center
    # --------------------------------------------------------

    current_cx = (left + right) / 2.0
    current_cy = (top + bottom) / 2.0

    target_cx = (width - 1) / 2.0
    target_cy = (height - 1) / 2.0

    shift_x = round(target_cx - current_cx)
    shift_y = round(target_cy - current_cy)

    # --------------------------------------------------------
    # Apply translation
    # --------------------------------------------------------

    result = [0] * (width * height)

    for y in range(height):
        for x in range(width):

            if rows[y][x] != 1:
                continue

            nx = x + shift_x
            ny = y + shift_y

            if (
                0 <= nx < width
                and
                0 <= ny < height
            ):
                result[ny * width + nx] = 1

    return result


# ============================================================
# 2. BASIC SHAPE FEATURES
# ============================================================

def iou_similarity(
    grid1: list[int],
    grid2: list[int]
) -> float:

    intersection = 0
    union = 0

    for a, b in zip(grid1, grid2):

        if a == 1 or b == 1:
            union += 1

        if a == 1 and b == 1:
            intersection += 1

    if union == 0:
        return 0.0

    return (intersection / union) * 100.0


def centroid_similarity(
    grid1: list[int],
    grid2: list[int]
) -> float:

    def centroid(grid):

        xs = []
        ys = []

        for idx, value in enumerate(grid):

            if value:

                xs.append(idx % GRID_W)
                ys.append(idx // GRID_W)

        if not xs:
            return None

        return (
            sum(xs) / len(xs),
            sum(ys) / len(ys)
        )

    c1 = centroid(grid1)
    c2 = centroid(grid2)

    if c1 is None or c2 is None:
        return 0.0

    distance = math.sqrt(
        (c1[0] - c2[0]) ** 2 +
        (c1[1] - c2[1]) ** 2
    )

    max_distance = math.sqrt(
        GRID_W ** 2 +
        GRID_H ** 2
    )

    return max(
        0.0,
        (1.0 - distance / max_distance) * 100.0
    )


def density_similarity(
    grid1: list[int],
    grid2: list[int]
) -> float:

    ink1 = sum(grid1)
    ink2 = sum(grid2)

    if ink1 == 0 and ink2 == 0:
        return 100.0

    if max(ink1, ink2) == 0:
        return 0.0

    return (
        min(ink1, ink2) /
        max(ink1, ink2)
    ) * 100.0


# ============================================================
# 3. OPENCV SHAPE DESCRIPTOR
# ============================================================

def grid_to_cv_image(grid: list[int]) -> np.ndarray:
    """
    Convert binary grid to OpenCV image.

    Ink = white
    Background = black
    """

    image = np.array(
        grid,
        dtype=np.uint8
    ).reshape(GRID_H, GRID_W)

    return image * 255


def largest_contour(image: np.ndarray):
    """
    Extract the largest meaningful external contour.

    Small isolated components/noise are ignored.
    The image must be a binary/grayscale OpenCV image.
    """

    if image is None:
        return None

    # Ensure proper binary representation.
    binary = np.where(
        image > 0,
        255,
        0
    ).astype(np.uint8)

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None

    # Remove extremely tiny components.
    meaningful = [
        contour
        for contour in contours
        if cv2.contourArea(contour) >= 4.0
    ]

    if not meaningful:
        return None

    return max(
        meaningful,
        key=cv2.contourArea
    )


def opencv_shape_similarity(
    grid1: list[int],
    grid2: list[int]
) -> float:
    """
    Compare two signature shapes using OpenCV contours.

    Pipeline:

        binary grid
            ↓
        4x nearest-neighbour scale
            ↓
        morphological closing
            ↓
        light dilation
            ↓
        external contour extraction
            ↓
        cv2.matchShapes()
            ↓
        bounded 0-100 similarity

    Lower matchShapes distance = better geometric match.
    """

    img1 = grid_to_cv_image(grid1)
    img2 = grid_to_cv_image(grid2)

    # --------------------------------------------------------
    # Scale up the tiny 64x32 signature.
    # --------------------------------------------------------

    img1_large = cv2.resize(
        img1,
        (GRID_W * 4, GRID_H * 4),
        interpolation=cv2.INTER_NEAREST
    )

    img2_large = cv2.resize(
        img2,
        (GRID_W * 4, GRID_H * 4),
        interpolation=cv2.INTER_NEAREST
    )

    # --------------------------------------------------------
    # Close tiny gaps without aggressively changing shape.
    # --------------------------------------------------------

    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (3, 3)
    )

    img1_closed = cv2.morphologyEx(
        img1_large,
        cv2.MORPH_CLOSE,
        close_kernel,
        iterations=1
    )

    img2_closed = cv2.morphologyEx(
        img2_large,
        cv2.MORPH_CLOSE,
        close_kernel,
        iterations=1
    )

    # --------------------------------------------------------
    # Light dilation.
    #
    # We only need enough thickness to prevent degenerate
    # 1-pixel contours. Too much dilation can merge separate
    # strokes and destroy shape information.
    # --------------------------------------------------------

    dilate_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (3, 3)
    )

    img1_processed = cv2.dilate(
        img1_closed,
        dilate_kernel,
        iterations=1
    )

    img2_processed = cv2.dilate(
        img2_closed,
        dilate_kernel,
        iterations=1
    )

    # --------------------------------------------------------
    # Extract meaningful contours.
    # --------------------------------------------------------

    contour1 = largest_contour(
        img1_processed
    )

    contour2 = largest_contour(
        img2_processed
    )

    if contour1 is None or contour2 is None:
        return 0.0

    # Degenerate contour protection.
    area1 = cv2.contourArea(contour1)
    area2 = cv2.contourArea(contour2)

    if area1 < 4.0 or area2 < 4.0:
        return 0.0

    try:

        # ----------------------------------------------------
        # OpenCV geometric shape comparison.
        #
        # I1 is already part of OpenCV's shape matching
        # implementation and uses Hu-invariant descriptors.
        # ----------------------------------------------------

        distance = float(
            cv2.matchShapes(
                contour1,
                contour2,
                cv2.CONTOURS_MATCH_I1,
                0.0
            )
        )

        if not math.isfinite(distance):
            return 0.0

        # ----------------------------------------------------
        # Convert OpenCV distance to 0-100 similarity.
        #
        # Keep this mapping deliberately conservative.
        # FAR/FRR calibration will determine the final threshold.
        # ----------------------------------------------------

        similarity = (
            100.0 *
            math.exp(-4.0 * distance)
        )

        return round(
            max(
                0.0,
                min(
                    100.0,
                    similarity
                )
            ),
            2
        )

    except Exception:
        return 0.0


def hu_moment_similarity(
    grid1: list[int],
    grid2: list[int]
) -> float:
    """
    Compare Hu-moment descriptors.

    FIX: Same scale-up + dilation as opencv_shape_similarity.
    Hu moments on zero-area contours are meaningless.
    """

    img1 = grid_to_cv_image(grid1)
    img2 = grid_to_cv_image(grid2)

    # Scale up and dilate before contour detection
    kernel = np.ones((3, 3), np.uint8)
    img1_large = cv2.resize(img1, (GRID_W * 4, GRID_H * 4), interpolation=cv2.INTER_NEAREST)
    img2_large = cv2.resize(img2, (GRID_W * 4, GRID_H * 4), interpolation=cv2.INTER_NEAREST)
    img1_dilated = cv2.dilate(img1_large, kernel, iterations=2)
    img2_dilated = cv2.dilate(img2_large, kernel, iterations=2)

    c1 = largest_contour(img1_dilated)
    c2 = largest_contour(img2_dilated)

    if c1 is None or c2 is None:
        return 0.0

    m1 = cv2.moments(c1)
    m2 = cv2.moments(c2)

    h1 = cv2.HuMoments(m1).flatten()
    h2 = cv2.HuMoments(m2).flatten()

    # Log transform makes the very small Hu values easier
    # to compare numerically.
    h1 = np.sign(h1) * np.log10(
        np.abs(h1) + 1e-30
    )

    h2 = np.sign(h2) * np.log10(
        np.abs(h2) + 1e-30
    )

    distance = float(
        np.mean(np.abs(h1 - h2))
    )

    similarity = 100.0 * math.exp(
        -0.35 * distance
    )

    return round(
        max(0.0, min(100.0, similarity)),
        2
    )


# ============================================================
# 4. COMPLETE VISUAL SHAPE SCORE
# ============================================================

def shape_similarity(
    grid1: list[int],
    grid2: list[int]
) -> dict:
    """
    Improved visual shape comparison for SignPay.

    This keeps the existing return structure so the rest of the
    backend does not need to change.

    Shape descriptor:

        15% IoU
        05% centroid
        05% density
        15% aspect-ratio / geometry
        15% local regional structure
        10% projection structure
        20% distance-map geometry
        10% OpenCV contour geometry
        05% Hu moments

    Important:
        OpenCV matchShapes() already uses Hu invariants internally,
        so OpenCV and explicit Hu are intentionally kept at modest
        weights rather than double-counting them.

    The goal is NOT exact pixel matching.

    The goal is to compare:
        - global geometry
        - local ink distribution
        - projection structure
        - geometric stroke proximity
        - contour shape

    Returns the same fields expected by the existing backend:
        score
        iou
        centroid
        density
        opencv
        hu
    """

    # ============================================================
    # 0. INPUT VALIDATION
    # ============================================================

    expected_size = GRID_W * GRID_H

    if (
        grid1 is None
        or grid2 is None
        or len(grid1) != expected_size
        or len(grid2) != expected_size
    ):
        return {
            "score": 0.0,
            "iou": 0.0,
            "centroid": 0.0,
            "density": 0.0,
            "opencv": 0.0,
            "hu": 0.0
        }

    # ------------------------------------------------------------
    # Convert to clean binary images.
    #
    # 0   = background
    # 255 = signature ink
    # ------------------------------------------------------------

    img1 = (
        np.asarray(grid1, dtype=np.uint8)
        .reshape(GRID_H, GRID_W)
    )

    img2 = (
        np.asarray(grid2, dtype=np.uint8)
        .reshape(GRID_H, GRID_W)
    )

    img1 = np.where(
        img1 > 0,
        255,
        0
    ).astype(np.uint8)

    img2 = np.where(
        img2 > 0,
        255,
        0
    ).astype(np.uint8)

    # ============================================================
    # 1. EXISTING BASIC FEATURES
    # ============================================================

    iou = iou_similarity(
        grid1,
        grid2
    )

    centroid = centroid_similarity(
        grid1,
        grid2
    )

    density = density_similarity(
        grid1,
        grid2
    )

    # ============================================================
    # 2. LIGHT MORPHOLOGICAL NORMALIZATION
    #
    # IMPORTANT:
    #
    # We deliberately avoid the previous heavy:
    #
    #     closing + dilation
    #
    # pipeline here.
    #
    # The signature has already been normalized to 64x32.
    # We only close tiny 1-pixel gaps.
    # ============================================================

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (3, 3)
    )

    proc1 = cv2.morphologyEx(
        img1,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=1
    )

    proc2 = cv2.morphologyEx(
        img2,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=1
    )

    # ============================================================
    # 3. GLOBAL GEOMETRY
    #
    # Compare:
    #
    #     width
    #     height
    #     aspect ratio
    #     occupied area
    #
    # This helps distinguish structurally different signatures
    # while still tolerating reasonable size variation.
    # ============================================================

    def geometry_descriptor(image):

        ys, xs = np.where(
            image > 0
        )

        if len(xs) == 0:
            return None

        x_min = int(xs.min())
        x_max = int(xs.max())

        y_min = int(ys.min())
        y_max = int(ys.max())

        width = max(
            1,
            x_max - x_min + 1
        )

        height = max(
            1,
            y_max - y_min + 1
        )

        area = float(
            np.count_nonzero(image)
        )

        aspect_ratio = (
            float(width) /
            float(height)
        )

        bbox_area = float(
            width * height
        )

        fill_ratio = (
            area / bbox_area
            if bbox_area > 0
            else 0.0
        )

        return {
            "width": float(width),
            "height": float(height),
            "aspect": aspect_ratio,
            "fill": fill_ratio
        }

    g1 = geometry_descriptor(proc1)
    g2 = geometry_descriptor(proc2)

    if g1 is None or g2 is None:

        geometry_score = 0.0

    else:

        # --------------------------------------------------------
        # Relative aspect-ratio difference.
        # --------------------------------------------------------

        aspect_diff = abs(
            g1["aspect"] -
            g2["aspect"]
        ) / max(
            g1["aspect"],
            g2["aspect"],
            1e-6
        )

        aspect_score = max(
            0.0,
            1.0 - aspect_diff
        ) * 100.0

        # --------------------------------------------------------
        # Width and height are intentionally given low influence.
        #
        # This allows natural size variation.
        # --------------------------------------------------------

        width_diff = abs(
            g1["width"] -
            g2["width"]
        ) / max(
            g1["width"],
            g2["width"],
            1.0
        )

        height_diff = abs(
            g1["height"] -
            g2["height"]
        ) / max(
            g1["height"],
            g2["height"],
            1.0
        )

        width_score = max(
            0.0,
            1.0 - width_diff
        ) * 100.0

        height_score = max(
            0.0,
            1.0 - height_diff
        ) * 100.0

        fill_diff = abs(
            g1["fill"] -
            g2["fill"]
        )

        fill_score = max(
            0.0,
            1.0 - fill_diff
        ) * 100.0

        geometry_score = (
            aspect_score * 0.50 +
            width_score * 0.15 +
            height_score * 0.15 +
            fill_score * 0.20
        )

    # ============================================================
    # 4. LOCAL REGIONAL SHAPE
    #
    # Divide the normalized signature into regions.
    #
    # This prevents two signatures with similar total ink amount
    # from receiving a high score merely because their overall
    # density is similar.
    #
    # Example:
    #
    #     +-----+-----+-----+-----+
    #     |     |     |     |     |
    #     +-----+-----+-----+-----+
    #     |     |     |     |     |
    #     +-----+-----+-----+-----+
    #
    # Each region contributes its normalized ink density.
    # ============================================================

    def regional_descriptor(image):

        rows = 2
        cols = 4

        values = []

        for r in range(rows):

            y0 = (
                r * GRID_H
            ) // rows

            y1 = (
                (r + 1) * GRID_H
            ) // rows

            for c in range(cols):

                x0 = (
                    c * GRID_W
                ) // cols

                x1 = (
                    (c + 1) * GRID_W
                ) // cols

                region = image[
                    y0:y1,
                    x0:x1
                ]

                if region.size == 0:
                    values.append(0.0)
                else:
                    values.append(
                        float(
                            np.count_nonzero(region)
                        ) /
                        float(region.size)
                    )

        return np.asarray(
            values,
            dtype=np.float32
        )

    r1 = regional_descriptor(proc1)
    r2 = regional_descriptor(proc2)

    regional_distance = float(
        np.mean(
            np.abs(r1 - r2)
        )
    )

    regional_score = max(
        0.0,
        1.0 - regional_distance
    ) * 100.0

    # ============================================================
    # 5. HORIZONTAL + VERTICAL PROJECTION STRUCTURE
    #
    # Instead of comparing exact pixels, compare how much ink
    # exists in each row and column.
    #
    # This captures structural differences such as:
    #
    #     lowercase vs uppercase
    #     different vertical distribution
    #     different horizontal concentration
    # ============================================================

    def projection_descriptor(image):

        binary = (
            image > 0
        ).astype(
            np.float32
        )

        horizontal = (
            np.sum(
                binary,
                axis=1
            ) /
            float(GRID_W)
        )

        vertical = (
            np.sum(
                binary,
                axis=0
            ) /
            float(GRID_H)
        )

        return (
            horizontal,
            vertical
        )

    h1, v1 = projection_descriptor(
        proc1
    )

    h2, v2 = projection_descriptor(
        proc2
    )

    horizontal_distance = float(
        np.mean(
            np.abs(h1 - h2)
        )
    )

    vertical_distance = float(
        np.mean(
            np.abs(v1 - v2)
        )
    )

    projection_score = (
        (
            max(
                0.0,
                1.0 - horizontal_distance
            ) * 100.0
        ) * 0.50
        +
        (
            max(
                0.0,
                1.0 - vertical_distance
            ) * 100.0
        ) * 0.50
    )

    # ============================================================
    # 6. DISTANCE-MAP SHAPE SIMILARITY
    #
    # This is the major new geometric feature.
    #
    # Instead of asking:
    #
    #     "Are the exact same pixels ON?"
    #
    # it asks:
    #
    #     "How far is the ink in A from the ink in B?"
    #
    # This makes the comparison more tolerant of natural
    # hand-drawn movement while still penalizing structural
    # displacement.
    #
    # OpenCV distanceTransform() computes distance to the nearest
    # zero/background pixel.
    # ============================================================

    def distance_map_similarity(
        binary1,
        binary2
    ):

        if (
            np.count_nonzero(binary1) == 0
            or
            np.count_nonzero(binary2) == 0
        ):
            return 0.0

        # --------------------------------------------------------
        # We want:
        #
        # distance from ink in image A to nearest ink in B.
        #
        # distanceTransform() measures distance to zero pixels,
        # therefore invert the target mask.
        # --------------------------------------------------------

        target1 = np.where(
            binary1 > 0,
            0,
            255
        ).astype(
            np.uint8
        )

        target2 = np.where(
            binary2 > 0,
            0,
            255
        ).astype(
            np.uint8
        )

        dist_to_1 = cv2.distanceTransform(
            target1,
            cv2.DIST_L2,
            3
        )

        dist_to_2 = cv2.distanceTransform(
            target2,
            cv2.DIST_L2,
            3
        )

        ink1_mask = (
            binary1 > 0
        )

        ink2_mask = (
            binary2 > 0
        )

        # --------------------------------------------------------
        # A -> B
        # --------------------------------------------------------

        d12_values = dist_to_2[
            ink1_mask
        ]

        # --------------------------------------------------------
        # B -> A
        # --------------------------------------------------------

        d21_values = dist_to_1[
            ink2_mask
        ]

        if (
            d12_values.size == 0
            or
            d21_values.size == 0
        ):
            return 0.0

        mean_d12 = float(
            np.mean(d12_values)
        )

        mean_d21 = float(
            np.mean(d21_values)
        )

        symmetric_distance = (
            mean_d12 +
            mean_d21
        ) / 2.0

        # --------------------------------------------------------
        # Maximum meaningful distance is approximately the
        # diagonal of the normalized canvas.
        # --------------------------------------------------------

        max_distance = math.sqrt(
            GRID_W ** 2 +
            GRID_H ** 2
        )

        normalized_distance = min(
            1.0,
            symmetric_distance /
            max_distance
        )

        return max(
            0.0,
            (
                1.0 -
                normalized_distance
            )
        ) * 100.0

    distance_score = distance_map_similarity(
        proc1,
        proc2
    )

    # ============================================================
    # 7. COMPLETE-CONTOUR OPEN-CV SHAPE
    #
    # IMPORTANT DIFFERENCE FROM THE OLD CODE:
    #
    # We DO NOT simply choose:
    #
    #     largest contour
    #
    # because that can throw away meaningful parts of a
    # signature.
    #
    # Instead:
    #
    #     - retrieve all external contours
    #     - ignore only tiny noise
    #     - compare every meaningful contour against the other
    #       signature
    #     - use a symmetric best-match aggregation
    #
    # This keeps disconnected meaningful parts from being
    # silently discarded.
    # ============================================================

    def meaningful_contours(image):

        contours, _ = cv2.findContours(
            image,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return []

        total_pixels = float(
            np.count_nonzero(image)
        )

        if total_pixels <= 0:
            return []

        result = []

        for contour in contours:

            area = float(
                cv2.contourArea(
                    contour
                )
            )

            perimeter = float(
                cv2.arcLength(
                    contour,
                    True
                )
            )

            # ----------------------------------------------------
            # Do not throw away useful thin signature components
            # merely because their contour area is small.
            #
            # Instead use perimeter and bounding-box size as
            # additional evidence.
            # ----------------------------------------------------

            x, y, w, h = cv2.boundingRect(
                contour
            )

            component_size = max(
                w * h,
                1
            )

            if (
                area < 1.0
                and
                perimeter < 4.0
            ):
                continue

            # Ignore microscopic noise.
            if (
                component_size <= 2
                and
                area < 1.0
            ):
                continue

            result.append(
                (
                    contour,
                    max(
                        area,
                        perimeter,
                        1.0
                    )
                )
            )

        return result

    contours1 = meaningful_contours(
        proc1
    )

    contours2 = meaningful_contours(
        proc2
    )

    def contour_match_score(
        contours_a,
        contours_b
    ):

        if not contours_a or not contours_b:
            return 0.0

        def one_direction(
            source,
            target
        ):

            weighted_scores = []
            weights = []

            for contour_a, weight_a in source:

                best_score = 0.0

                for contour_b, _ in target:

                    try:

                        distance = float(
                            cv2.matchShapes(
                                contour_a,
                                contour_b,
                                cv2.CONTOURS_MATCH_I2,
                                0.0
                            )
                        )

                    except Exception:

                        continue

                    if not math.isfinite(
                        distance
                    ):
                        continue

                    # ------------------------------------------------
                    # Convert contour distance to bounded similarity.
                    #
                    # I2 is a Hu-based contour distance.
                    # Lower = more similar.
                    # ------------------------------------------------

                    score = (
                        100.0 *
                        math.exp(
                            -3.0 *
                            distance
                        )
                    )

                    score = max(
                        0.0,
                        min(
                            100.0,
                            score
                        )
                    )

                    if score > best_score:
                        best_score = score

                weighted_scores.append(
                    best_score
                )

                weights.append(
                    weight_a
                )

            if not weighted_scores:
                return 0.0

            return (
                sum(
                    score * weight
                    for score, weight
                    in zip(
                        weighted_scores,
                        weights
                    )
                )
                /
                max(
                    sum(weights),
                    1e-6
                )
            )

        forward = one_direction(
            contours_a,
            contours_b
        )

        backward = one_direction(
            contours_b,
            contours_a
        )

        return (
            forward +
            backward
        ) / 2.0

    opencv_score = contour_match_score(
        contours1,
        contours2
    )

    # ============================================================
    # 8. HU MOMENT SUPPORTING SCORE
    #
    # We retain Hu moments, but deliberately keep them low-weight.
    #
    # They are already used internally by matchShapes().
    # ============================================================

    def hu_score_for_image(image):

        contours = meaningful_contours(
            image
        )

        if not contours:
            return None

        # Use the union of meaningful contour geometry through
        # raster moments rather than selecting only the largest
        # contour.
        binary = (
            image > 0
        ).astype(
            np.uint8
        )

        moments = cv2.moments(
            binary,
            binaryImage=True
        )

        if abs(
            float(
                moments.get(
                    "m00",
                    0.0
                )
            )
        ) < 1e-9:

            return None

        hu = cv2.HuMoments(
            moments
        ).flatten()

        # Stable signed log representation.
        hu = (
            np.sign(hu) *
            np.log10(
                np.abs(hu) +
                1e-30
            )
        )

        return hu

    hu1 = hu_score_for_image(
        proc1
    )

    hu2 = hu_score_for_image(
        proc2
    )

    if hu1 is None or hu2 is None:

        hu_score = 0.0

    else:

        hu_distance = float(
            np.mean(
                np.abs(
                    hu1 -
                    hu2
                )
            )
        )

        hu_score = (
            100.0 *
            math.exp(
                -0.30 *
                hu_distance
            )
        )

        hu_score = max(
            0.0,
            min(
                100.0,
                hu_score
            )
        )

    # ============================================================
    # 9. FINAL SHAPE FUSION
    #
    # IMPORTANT:
    #
    # We intentionally do NOT give OpenCV 35% anymore.
    #
    # The shape score should represent several complementary
    # properties instead of allowing one Hu-based contour
    # descriptor to dominate.
    #
    #                         Weight
    # ------------------------------------------------------------
    # IoU                       15%
    # Centroid                   5%
    # Density                    5%
    # Geometry                  15%
    # Regional structure        15%
    # Projection structure      10%
    # Distance geometry         20%
    # OpenCV contour            10%
    # Hu moments                5%
    # ------------------------------------------------------------
    # Total                    100%
    # ============================================================

    score = (
        iou * 0.15
        +
        centroid * 0.05
        +
        density * 0.05
        +
        geometry_score * 0.15
        +
        regional_score * 0.15
        +
        projection_score * 0.10
        +
        distance_score * 0.20
        +
        opencv_score * 0.10
        +
        hu_score * 0.05
    )

    score = max(
        0.0,
        min(
            100.0,
            score
        )
    )

    return {
        "score": round(
            score,
            2
        ),

        "iou": round(
            iou,
            2
        ),

        "centroid": round(
            centroid,
            2
        ),

        "density": round(
            density,
            2
        ),

        "opencv": round(
            opencv_score,
            2
        ),

        "hu": round(
            hu_score,
            2
        )
    }


# ============================================================
# 5. BEHAVIOR DESCRIPTOR
# ============================================================

def extract_behavior_descriptor(
    timing: list[dict]
) -> dict:
    """
    Convert raw canvas events into behavioral features.

    Features:
      - stroke count
      - total duration
      - stroke durations
      - pauses
      - normalized trajectory
      - velocity profile
      - direction sequence
    """

    points = [
        p for p in timing
        if "x" in p and "y" in p
    ]

    if len(points) < 3:

        return {
            "valid": False,
            "stroke_count": 0,
            "duration": 0.0,
            "stroke_durations": [],
            "pauses": [],
            "trajectory": [],
            "velocity": [],
            "directions": []
        }

    xs = [float(p["x"]) for p in points]
    ys = [float(p["y"]) for p in points]

    min_x = min(xs)
    max_x = max(xs)
    min_y = min(ys)
    max_y = max(ys)

    range_x = max_x - min_x or 1.0
    range_y = max_y - min_y or 1.0

    trajectory = [
        (
            (float(p["x"]) - min_x) / range_x,
            (float(p["y"]) - min_y) / range_y
        )
        for p in points
    ]

    timestamps = [
        float(p["t"])
        for p in points
    ]

    duration = max(
        1.0,
        timestamps[-1] - timestamps[0]
    )

    velocities = []
    directions = []

    for i in range(1, len(points)):

        dt = (
            float(points[i]["t"]) -
            float(points[i - 1]["t"])
        )

        dt = max(dt, 1.0)

        dx = (
            float(points[i]["x"]) -
            float(points[i - 1]["x"])
        )

        dy = (
            float(points[i]["y"]) -
            float(points[i - 1]["y"])
        )

        distance = math.sqrt(
            dx * dx +
            dy * dy
        )

        velocities.append(
            distance / dt
        )

        angle = math.atan2(dy, dx)

        directions.append(angle)

    # Stroke information comes from frontend stroke_id.
    stroke_ids = [
        p.get("stroke_id", 0)
        for p in points
    ]

    stroke_count = (
        len(set(stroke_ids))
        if stroke_ids
        else 1
    )

    stroke_durations = []

    for sid in sorted(set(stroke_ids)):

        stroke_points = [
            p for p in points
            if p.get("stroke_id", 0) == sid
        ]

        if len(stroke_points) >= 2:

            d = (
                float(stroke_points[-1]["t"]) -
                float(stroke_points[0]["t"])
            )

            stroke_durations.append(
                max(0.0, d)
            )

    # Direction sequence compressed into 8 sectors.
    direction_bins = []

    for angle in directions:

        sector = int(
            ((angle + math.pi) /
             (2 * math.pi)) * 8
        ) % 8

        direction_bins.append(sector)

    return {
        "valid": True,
        "stroke_count": stroke_count,
        "duration": duration,
        "stroke_durations": stroke_durations,
        "pauses": [],
        "trajectory": trajectory,
        "velocity": velocities,
        "directions": direction_bins
    }


def sequence_similarity(
    seq1,
    seq2
) -> float:

    if not seq1 or not seq2:
        return 50.0

    n = min(len(seq1), len(seq2))

    if n == 0:
        return 50.0

    matches = sum(
        1
        for a, b in zip(
            seq1[:n],
            seq2[:n]
        )
        if a == b
    )

    length_penalty = (
        min(len(seq1), len(seq2)) /
        max(len(seq1), len(seq2))
    )

    return (
        (matches / n) *
        100.0 *
        0.8
        +
        length_penalty *
        100.0 *
        0.2
    )


def trajectory_similarity(
    t1,
    t2
) -> float:

    if len(t1) < 3 or len(t2) < 3:
        return 50.0

    n = 30

    def sample(seq):

        result = []

        for i in range(n):

            idx = int(
                i * (len(seq) - 1) /
                (n - 1)
            )

            result.append(seq[idx])

        return result

    a = sample(t1)
    b = sample(t2)

    total = 0.0

    for p, q in zip(a, b):

        dx = p[0] - q[0]
        dy = p[1] - q[1]

        total += math.sqrt(
            dx * dx +
            dy * dy
        )

    average = total / n

    return max(
        0.0,
        100.0 -
        min(100.0, average / 1.414 * 100.0)
    )


def velocity_similarity(
    v1,
    v2
) -> float:

    if len(v1) < 2 or len(v2) < 2:
        return 50.0

    n = min(20, len(v1), len(v2))

    def sample(values):

        return [
            values[
                int(i * (len(values) - 1) /
                    (n - 1))
            ]
            for i in range(n)
        ]

    a = sample(v1)
    b = sample(v2)

    max_velocity = max(
        max(a),
        max(b),
        1e-9
    )

    total = 0.0

    for x, y in zip(a, b):

        total += abs(
            x - y
        ) / max_velocity

    average = total / n

    return max(
        0.0,
        100.0 -
        min(100.0, average * 100.0)
    )


def duration_similarity(
    d1: float,
    d2: float
) -> float:

    if d1 <= 0 or d2 <= 0:
        return 50.0

    ratio = min(d1, d2) / max(d1, d2)

    return ratio * 100.0


def behavior_similarity(
    timing1: list[dict],
    timing2: list[dict]
) -> dict:
    """
    Compare behavioral descriptors.

    This is deliberately separate from image shape.
    """

    b1 = extract_behavior_descriptor(timing1)
    b2 = extract_behavior_descriptor(timing2)

    if not b1["valid"] or not b2["valid"]:

        return {
            "score": 0.0,
            "trajectory": 0.0,
            "velocity": 0.0,
            "direction": 0.0,
            "duration": 0.0,
            "stroke": 0.0
        }

    trajectory = trajectory_similarity(
        b1["trajectory"],
        b2["trajectory"]
    )

    velocity = velocity_similarity(
        b1["velocity"],
        b2["velocity"]
    )

    direction = sequence_similarity(
        b1["directions"],
        b2["directions"]
    )

    duration = duration_similarity(
        b1["duration"],
        b2["duration"]
    )

    stroke = (
        100.0
        if b1["stroke_count"] ==
           b2["stroke_count"]
        else
        max(
            0.0,
            100.0 -
            abs(
                b1["stroke_count"] -
                b2["stroke_count"]
            ) * 25.0
        )
    )

    score = (
        trajectory * 0.30 +
        velocity * 0.20 +
        direction * 0.25 +
        duration * 0.10 +
        stroke * 0.15
    )

    return {
        "score": round(score, 2),
        "trajectory": round(trajectory, 2),
        "velocity": round(velocity, 2),
        "direction": round(direction, 2),
        "duration": round(duration, 2),
        "stroke": round(stroke, 2)
    }


# ============================================================
# 6. FINAL DECISION ENGINE
# ============================================================

def compare_signatures(
    enrolled: dict,
    test_grid: list[int],
    test_timing: list[dict]
) -> dict:

    shape = shape_similarity(
        enrolled["grid"],
        test_grid
    )

    behavior = behavior_similarity(
        enrolled["timing"],
        test_timing
    )

    final_score = (
        shape["score"] * SHAPE_WEIGHT +
        behavior["score"] * BEHAVIOR_WEIGHT
    )

    final_score = round(
        final_score,
        2
    )

    matched = (
        final_score >= THRESHOLD
    )

    return {
        "matched": matched,
        "score": final_score,

        "shape_score": shape["score"],
        "behavior_score": behavior["score"],

        "timing_score": behavior["score"],

        "shape": shape,
        "behavior": behavior,

        "threshold": THRESHOLD,

        "verdict":
            "AUTHENTICATED ✓"
            if matched
            else
            "REJECTED ✗"
    }


# ============================================================
# 7. .SGPX STORAGE
# ============================================================

def save_sgpx(
    user_id: str,
    grid: list[int],
    timing: list[dict],
    ink_count: int
) -> str:

    grid_str = ''.join(
        map(str, grid)
    )

    fingerprint = hashlib.sha256(
        grid_str.encode()
    ).hexdigest()[:32]

    sgpx_data = {
        "format": "sgpx",
        "version": "2.0",
        "user_id": user_id,
        "enrolled_at": time.time(),

        "grid_width": GRID_W,
        "grid_height": GRID_H,

        "ink_count": ink_count,

        "grid": grid,
        "timing": timing,

        "fingerprint": fingerprint,

        "never_transmit": True
    }

    filepath = os.path.join(
        STORAGE_DIR,
        f"{user_id}.sgpx"
    )

    with open(
        filepath,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            sgpx_data,
            f,
            indent=2
        )

    return filepath


def load_sgpx(
    user_id: str
) -> dict | None:

    filepath = os.path.join(
        STORAGE_DIR,
        f"{user_id}.sgpx"
    )

    if not os.path.exists(filepath):
        return None

    with open(
        filepath,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)


def delete_sgpx(
    user_id: str
) -> bool:

    filepath = os.path.join(
        STORAGE_DIR,
        f"{user_id}.sgpx"
    )

    if os.path.exists(filepath):

        os.remove(filepath)
        return True

    return False


def has_enough_ink(
    grid: list[int]
) -> bool:

    return sum(grid) > 50


def get_sgpx_summary(
    user_id: str
) -> str:

    data = load_sgpx(user_id)

    if not data:
        return "No signature enrolled"

    enrolled_time = time.strftime(
        "%Y-%m-%d %H:%M:%S",
        time.localtime(
            data["enrolled_at"]
        )
    )

    return (
        f"[.sgpx v{data['version']}]\n"
        f"user: {data['user_id']}\n"
        f"enrolled: {enrolled_time}\n"
        f"ink_count: {data['ink_count']}\n"
        f"timing_points: "
        f"{len(data['timing'])}\n"
        f"fingerprint: "
        f"{data['fingerprint']}\n"
        f"never_transmit: "
        f"{data['never_transmit']}"
    )
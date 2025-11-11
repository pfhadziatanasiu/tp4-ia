import cv2
import numpy as np

def detect_main_circle(
    image_path: str = "engine.png",
    known_radius_px: int | None = None,
    radius_tolerance_px: int = 6,
    dp: float = 1.2,
    param1: int = 120,
    param2: int = 25
):
    """
    Detect the largest/central circle from a black-on-white technical drawing
    using the Hough Circle Transform.

    Args:
        image_path: Path to the input image (PNG/JPG).
        known_radius_px: If you already know the circle radius in pixels,
                         set this to force a tight radius search window.
        radius_tolerance_px: +/- tolerance around the known radius (in px).
        dp: Inverse ratio of the accumulator resolution to the image resolution.
            (1.2 = accumulator has ~1/1.2 the resolution of the image)
        param1: High threshold for the internal Canny edge detector.
        param2: Accumulator threshold for circle centers.
                (Lower -> more detections; Higher -> fewer, stronger detections)

    Returns:
        (overlay_bgr, (x0, y0, r)) where:
            overlay_bgr: BGR image with the detected circle drawn.
            (x0, y0, r): center coordinates and radius in pixels (ints).
    """
    # --- 1) Load image (BGR) ---
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    # --- 2) Convert to grayscale ---
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # --- 3) Pre-filter: light denoise + contrast normalization ---
    # Median blur preserves sharp edges better than Gaussian on line drawings.
    gray = cv2.medianBlur(gray, 5)

    # Optional: improve contrast of thin black lines on white background
    # (Uncomment if your image looks low-contrast)
    # gray = cv2.equalizeHist(gray)

    H, W = gray.shape[:2]
    min_side = min(H, W)

    # --- 4) Define radius search window ---
    if known_radius_px is not None:
        minR = max(1, int(known_radius_px - radius_tolerance_px))
        maxR = int(known_radius_px + radius_tolerance_px)
        minDist = int(min_side * 0.25)  # distance between centers (not critical if we want only one)
    else:
        # If radius is unknown, scan a sensible window around the central wheel size.
        # You can tighten these percentages if needed.
        minR = int(min_side * 0.15)
        maxR = int(min_side * 0.45)
        minDist = int(min_side * 0.20)

    # --- 5) Run Hough Circle Transform ---
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=dp,
        minDist=minDist,
        param1=param1,     # Canny high threshold (low is param1/2 internally)
        param2=param2,     # Accumulator threshold (peak threshold)
        minRadius=minR,
        maxRadius=maxR
    )

    overlay = img.copy()
    if circles is None:
        print("No circles were found. Try lowering param2 or widening (minRadius, maxRadius).")
        return overlay, None

    # Keep the largest circle (central wheel is typically the largest in this drawing)
    circles = np.uint16(np.around(circles[0]))
    circles_sorted = sorted(circles.tolist(), key=lambda c: c[2], reverse=True)
    x0, y0, r = circles_sorted[0]  # (center_x, center_y, radius)

    # --- 6) Draw result ---
    # Outer circle
    cv2.circle(overlay, (x0, y0), r, (0, 255, 0), 2)
    # Center mark
    cv2.circle(overlay, (x0, y0), 2, (0, 0, 255), 3)

    # --- 7) Report ---
    print(f"Detected center (x0, y0) = ({x0}, {y0}), radius r = {r} px")
    print(f"Search window used: minRadius={minR}, maxRadius={maxR}, dp={dp}, param1={param1}, param2={param2}")
    print("The origin (0, 0) is located at the top-left corner of the image")
    print("The x-axis increases to the right")
    print("The y-axis increases downwards")

    return overlay, (int(x0), int(y0), int(r))


if __name__ == "__main__":
    # If you know the pixel radius already, set known_radius_px=<value> (and adjust tolerance).
    # Otherwise leave it as None to auto-scan.
    overlay_img, circle = detect_main_circle(
        image_path="engine.png",
        known_radius_px=97,     # e.g., set to 200 if you measured it in pixels
        radius_tolerance_px=6,
        dp=1.2,
        param1=120,
        param2=25
    )

    # Show results
    cv2.imshow("Detected Circle (overlay)", overlay_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Optionally save the overlay image:
    cv2.imwrite("engine_circle_overlay.png", overlay_img)
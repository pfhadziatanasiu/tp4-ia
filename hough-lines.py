import cv2
import numpy as np

# 1) Load the input image
# Replace "image.png" with your real image file
img = cv2.imread("engine.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 2) Smooth the image to reduce noise
# Gaussian blur avoids detecting false edges caused by noise
gray = cv2.GaussianBlur(gray, (5, 5), 1)

# 3) Edge detection (Canny)
# The thresholds (150,220) may need tuning depending on the lighting and contrast
edges = cv2.Canny(gray, 150, 220)

# 4) Apply the Hough Transform for line detection
# rho = 1 pixel resolution
# theta = 1 degree resolution (π/180 radians)
# threshold = minimum number of votes needed to consider something a line
lines = cv2.HoughLines(edges, rho=1, theta=np.pi/180, threshold=245)

# 5) Draw the detected lines on a copy of the original image
result = img.copy()

if lines is not None:
    for line in lines:
        rho, theta = line[0]  # Extract line parameters

        # Convert from (rho, theta) to Cartesian line representation
        a = np.cos(theta)
        b = np.sin(theta)

        # (x0, y0) is the closest point on the line to the origin
        x0 = a * rho
        y0 = b * rho

        # Create two far points along the line direction so the line spans the entire image
        # The direction of the line is perpendicular to the normal vector (−b, a)
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * ( a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * ( a))

        # Draw the line in green
        cv2.line(result, (x1, y1), (x2, y2), (0, 255, 0), 2)

# 6) Display the results
cv2.imshow("Original Image", img)
cv2.imshow("Edge Map (Canny)", edges)
cv2.imshow("Detected Lines (Hough)", result)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 7) Save the results
# Save the result image to a file
cv2.imwrite("engine_edges.png", edges)
cv2.imwrite("engine_lines_overlay.png", result)

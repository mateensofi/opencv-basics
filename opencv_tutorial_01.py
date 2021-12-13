# Imports
import imutils
import cv2

# Read the image and show its dimensions
image = cv2.imread("./images/jp.png")
(h, w, d) = image.shape
print("Width={}, Height={}, Depth={}".format(w, h, d))

cv2.imshow("Jurassic Park Image", image)
cv2.waitKey(0)

# Assess the RGB pixel located at x=50, y=100
(B, G, R) = image[100, 50]
print("R={}, G={}, B={}".format(R, G, B))

# Extract a 100x100 pixel square ROI (Region of Interest)
# from the input image starting at x=320,y=60
# at ending at x=420,y=160
roi = image[60:160, 320:420]
cv2.imshow("Region of Interest", roi)
cv2.waitKey(0)

resized = cv2.resize(image, (200, 200))
cv2.imshow("Resized Image", resized)
cv2.waitKey(0)

# Fix resizing and distorted aspect ratio
# Let's resize the width to be 300px but compute
# the new height based on the aspect ratio
r = 300.0 / w
dim = (300, int(h * r))
resized = cv2.resize(image, dim)
cv2.imshow("Resized Fixed Aspect Ratio", resized)
cv2.waitKey(0)


# Manually computing the aspect ratio can be a
# pain so let's use the imutils library instead
resized = imutils.resize(image, width=300)
cv2.imshow("Imutils Resized Image", resized)
cv2.waitKey(0)

# Let's rotate an image 45 degrees clockwise using OpenCV by first
# computing the image center, then constructing the rotation matrix,
# and then finally applying the affine warp
center = (w // 2, h // 2)
M = cv2.getRotationMatrix2D(center, angle=-45, scale=1.0)
rotated = cv2.warpAffine(image, M, (w, h))
cv2.imshow("OpenCV Rotation", rotated)
cv2.waitKey(0)

# Rotating the matrix using imutils with less code
rotated = imutils.rotate(image, angle=-45)
cv2.imshow("Imutils Rotation", rotated)
cv2.waitKey(0)

# OpenCV doesn't "care" if our rotated image is clipped after rotation,
# so we can instead use another imutils convenience function to help
# us out
rotated = imutils.rotate_bound(image, angle=45)
cv2.imshow("Imutils Rotate Bound", rotated)
cv2.waitKey(0)

# Apply a Gaussian Blur with 11x11 kernel to smoothen it
blurred = cv2.GaussianBlur(src=image, ksize=(11, 11), sigmaX=0)
cv2.imshow("Gaussian Blurred Image", blurred)
cv2.waitKey(0)

# Draw a 2px thick red rectangle around the face of Ian Malcom
output = image.copy()
cv2.rectangle(output, pt1=(320, 60), pt2=(420, 160), color=(0, 0, 255), thickness=2)
cv2.imshow("Rectangle around face", output)
cv2.waitKey(0)


# Draw a blue 20px (filled in) circle on the image centered at x=300, y=150
output = image.copy()
cv2.circle(output, center=(300, 150), radius=20, color=(255, 0, 0), thickness=-1)
cv2.imshow("Blue Circle", output)
cv2.waitKey(0)


# Draw a 5px thick red line from x=60,y=20 to x=400,y=200
output = image.copy()
cv2.line(output, pt1=(60, 20), pt2=(400, 200), color=(0, 0, 255), thickness=5)
cv2.imshow("Red Line", output)
cv2.waitKey(0)

# Draw green text on the image
output = image.copy()
cv2.putText(output, "OpenCV + Jurassic Park!", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
cv2.imshow("Text", output)
cv2.waitKey(0)

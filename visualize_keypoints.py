import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

# Set paths
image_path = r"E:\10.png"
points_path = r"E:\10.npy"
print(f"Checking image path: {os.path.exists(image_path)}")
print(f"Checking points path: {os.path.exists(points_path)}")

# Read image
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
if image is None:
    raise ValueError(f"Failed to read image: {image_path}")

# Read keypoints
try:
    points = np.load(points_path)
except Exception as e:
    raise ValueError(f"Failed to read points file: {points_path}, error: {str(e)}")

print(f"Image shape: {image.shape}")
print(f"Points shape: {points.shape}")
print("\nKeypoint coordinates:")
for i, (x, y) in enumerate(points):
    print(f"Point {i+1}: ({x:.2f}, {y:.2f})")

# Create visualization
plt.figure(figsize=(15, 5))

# Show original image with keypoints
plt.subplot(131)
plt.imshow(image, cmap='gray')
plt.plot(points[:, 1], points[:, 0], 'r.', markersize=10, label='Original')
plt.title('Image with Keypoints')
plt.grid(True)
plt.legend()

# Show image with numbered keypoints
plt.subplot(132)
plt.imshow(image, cmap='gray')
for i, (x, y) in enumerate(points):
    plt.plot(y, x, 'r.', markersize=10)
    plt.text(y+2, x+2, str(i+1), color='yellow', fontsize=8)
plt.title('Numbered Keypoints')
plt.grid(True)

# Generate heatmap
heatmap = np.zeros_like(image, dtype=np.float32)
for x, y in points:
    # Add Gaussian distribution at each keypoint
    x, y = int(round(x)), int(round(y))
    sigma = 3
    size = 6 * sigma + 1
    x_range = np.arange(max(0, x - 3*sigma), min(image.shape[1], x + 3*sigma + 1))
    y_range = np.arange(max(0, y - 3*sigma), min(image.shape[0], y + 3*sigma + 1))
    X, Y = np.meshgrid(x_range, y_range)
    gaussian = np.exp(-((X - x)**2 + (Y - y)**2)/(2*sigma**2))
    heatmap[Y.min():Y.max()+1, X.min():X.max()+1] += gaussian

# Show heatmap
plt.subplot(133)
plt.imshow(heatmap, cmap='jet')
plt.title('Keypoint Heatmap')
plt.grid(True)

plt.tight_layout()
plt.savefig('keypoint_visualization1.png', dpi=300, bbox_inches='tight')
print("\nVisualization saved to: keypoint_visualization.png")
plt.close() 
plt.close() 
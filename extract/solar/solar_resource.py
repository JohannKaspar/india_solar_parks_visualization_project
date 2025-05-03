import cv2
import numpy as np
from collections import Counter
from matplotlib import pyplot as plt
import pandas as pd
from scipy.interpolate import griddata

def main():
    # 1. Load the image
    img = cv2.imread('extract/solar/india_solar_resource_map.png')
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 2. Manually define legend colors and their GHI values (from the legend bar)
    # These RGB values should be sampled from the legend in the image
    legend = {
        (255, 255, 153): 3.0,   # light yellow
        (255, 204, 102): 3.5,   # light orange
        (255, 153, 51): 4.0,    # orange
        (255, 102, 0): 4.5,     # dark orange
        (255, 51, 0): 5.0,      # red-orange
        (204, 0, 0): 5.5,       # red
        (153, 0, 0): 6.0,       # dark red
        (67, 67, 67): np.nan,       # dark red
    }

    # 3. Function to find closest legend color
    def closest_color(pixel, legend_colors):
        color, dist = min(
            ((color, np.linalg.norm(np.array(pixel) - np.array(color))) for color in legend_colors),
            key=lambda t: t[1]
        )
        return color, dist

    # 4. Classify each pixel
    height, width, _ = img_rgb.shape
    output = np.zeros((height, width))
    # Create a 5x5 kernel for averaging
    kernel = np.ones((5, 5), np.float32) / 25
    
    # Apply convolution to smooth the image
    smoothed = cv2.filter2D(img_rgb, -1, kernel)
    
    # Map bounds (India, as per the map)
    min_lon, max_lon = 66.3, 98
    min_lat, max_lat = 2, 41

    # Process every 5th pixel for efficiency
    rows = []

    for y in range(0, height, 2):
        for x in range(0, width, 2):
            pixel = tuple(smoothed[y, x])
            color, dist = closest_color(pixel, legend.keys())
            ghi = legend[color]
            lon = min_lon + (x / (width - 1)) * (max_lon - min_lon)
            lat = max_lat - (y / (height - 1)) * (max_lat - min_lat)
            rows.append({'x': x, 'y': y, 'lat': lat, 'lon': lon, 'ghi': ghi})

    # Save as CSV
    df = pd.DataFrame(rows)
    df.to_csv('extract/solar/extracted_ghi_latlon.csv', index=False)

    # Optional: Visualize
    plt.scatter(df['lon'], df['lat'], c=df['ghi'], cmap='hot', s=1)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Extracted GHI (lat/lon)')
    plt.colorbar(label='GHI (kWh/m²/day)')
    plt.show()

if __name__ == "__main__":
    main()
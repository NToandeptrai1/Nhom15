import urllib.request
import os

def download_file(url, filename):
    print(f"Downloading {url}...")
    urllib.request.urlretrieve(url, filename)
    print(f"Saved to {filename} (Size: {os.path.getsize(filename)} bytes)")

models_dir = "models"
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

# Reliable URLs (hopefully)
urls = {
    "SSD_MobileNet.prototxt": "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/voc/MobileNetSSD_deploy.prototxt",
    "SSD_MobileNet.caffemodel": "https://github.com/chuanqi305/MobileNet-SSD/raw/master/MobileNetSSD_deploy.caffemodel"
}

for name, url in urls.items():
    path = os.path.join(models_dir, name)
    try:
        download_file(url, path)
    except Exception as e:
        print(f"Failed to download {name}: {e}")

print("Done.")

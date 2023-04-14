# all `import` at the beginning
import json
import base64
import cv2
import numpy as np
import requests
import time

# --- constants ---  

ROBOFLOW_API_KEY = '1Lh044JbccYkat1iYLCB'
ROBOFLOW_MODEL = 'product_finder/1'
ROBOFLOW_SIZE = 400

# --- functions ---

params = {
    "api_key": ROBOFLOW_API_KEY,
    "format": "image",
    "stroke": "5"
}

headers = {
    "Content-Type": "application/x-www-form-urlencoded"
}

url = f"https://detect.roboflow.com/{ROBOFLOW_MODEL}"


def infer(img):
    # Resize (while maintaining the aspect ratio) to improve speed and save bandwidth
    height, width, channels = img.shape
    scale = ROBOFLOW_SIZE / max(height, width)
    img = cv2.resize(img, (round(scale * width), round(scale * height)))

    # Encode image to base64 string
    retval, buffer = cv2.imencode('.jpg', img)
    img_str = base64.b64encode(buffer)

    # Get prediction from Roboflow Infer API
    response = requests.post(url, params=params, data=img_str, headers=headers, stream=True)
    data = response.raw.read()
    
    #print(response.request.url)
    
    if not response.ok:
        print('status:', response.status_code)
        print('data:', data)
        return
    
    # Parse result image
    image = np.asarray(bytearray(data), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    return image

# --- main ---

video = cv2.VideoCapture(0)

while True:

    start = time.time()

    ret, img = video.read()
    
    if ret:
        
        image = infer(img)
        
        if image is not None:
            cv2.imshow('image', image)
        
            if cv2.waitKey(1) == ord('q'):  # `waitKey` should be after `imshow()` - to update image in window and to get key from window
                break
        
            end = time.time()
            print( 1/(end-start), "fps")  # print() automatically add space between elements - you don't need space in "fps"
        
# - end -

video.release()
cv2.destroyAllWindows()
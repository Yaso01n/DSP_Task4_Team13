from flask import Flask, render_template, request , jsonify
import numpy as np
from scipy.fft import fft2 ,ifft2
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import io 
import base64 
import seaborn as sns
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import plotly.graph_objs as go
import plotly.express as px
import skimage.io 
from skimage.color import rgb2gray
from image import Processing

app = Flask(__name__,template_folder="templates")

def write_file_to_image(file):
    npimg = np.fromstring(file, np.uint8)
    img = cv2.imdecode(npimg,cv2.IMREAD_COLOR)
    img = Image.fromarray(img.astype("uint8"))
    rawBytes = io.BytesIO()
    img.save(rawBytes, "JPEG")
    rawBytes.seek(0)
    img_base64 = base64.b64encode(rawBytes.read())
    with open("image.jpg", "wb") as fh:
        fh.write(base64.decodebytes(img_base64)) 

def savefigures(magnitude_spectrum,mag_name,phase_spectrum,phase_name):
    fig =plt.figure(figsize=(15, 20))
    plt.imshow(magnitude_spectrum, cmap='gray')
    plt.savefig(mag_name)
    plt.imshow(phase_spectrum, cmap='gray')
    plt.savefig(phase_name)      
    im1= Image.open(mag_name)
    im2= Image.open(phase_name) 
    data = io.BytesIO()
    im1.save(data, "JPEG")
    data2 = io.BytesIO()
    im2.save(data2, "JPEG")
    encoded_img_data1 = base64.b64encode(data.getvalue())
    encoded_img_data2=base64.b64encode(data2.getvalue())
    return encoded_img_data1,encoded_img_data2  


def save_output(img_output,name):
    fig =plt.figure(figsize=(15, 20))
    fig.patch.set_facecolor('#ffffff')
    desert_coffee_shift = img_output + img_output.min()
    desert_coffee_shift[desert_coffee_shift>255] = 255
    img_output[img_output>255] = 255
    img_output[img_output <0] = 0
    plt.imshow(img_output ,cmap='gray')
    fig =plt.figure(figsize=(15, 20))
    plt.imshow( img_output, cmap='gray')
    plt.savefig(name)
    im3= Image.open(name)
    data3 = io.BytesIO()
    im3.save(data3, "JPEG")
    encoded_img_data3=base64.b64encode(data3.getvalue())
    return encoded_img_data3   


data=[]
images=[]

@app.route('/', methods=['POST','GET'])
def home():
    if request.method == "POST":
        file= request.files['image'].read()   # /// read image
        default_value = '0'
        name = request.form.get('name', default_value)   # //// to know which image is sent (image1 or iamge2)
        write_file_to_image(file)  
        img = skimage.io.imread("image.jpg") 
        if name==str(1):
            cv2.imwrite('image1.jpg', img)  
            image1_object =Processing('image1.jpg')
            data.append(image1_object.magnitude)
            data.append(image1_object.phase)
            encoded_img_data1,encoded_img_data2= savefigures(image1_object.magnitude_spectrum,"mag1.jpg",image1_object.phase_spectrum,"phase1.jpg")
            image1_object.combine(image1_object.magnitude,image1_object.phase)
            rec=save_output(image1_object.img_output,"rec.jpg")
        else :
            cv2.imwrite('image2.jpg', img)
            image2_object=Processing('image2.jpg')
            data.append(image2_object.magnitude)
            data.append(image2_object.phase)
            encoded_img_data1,encoded_img_data2=savefigures(image2_object.magnitude_spectrum,"mag2.jpg",image2_object.phase_spectrum,"phase2.jpg")
            image2_object.combine(data[0],data[3])
            rec=save_output(image2_object.img_output,"rec1.jpg")
            data.clear()
        return jsonify({'status':str(encoded_img_data1),'status2':str(encoded_img_data2),'status3':str(rec)})     
    return render_template('test.html')


if __name__ == '__main__':
   app.run(debug=True)
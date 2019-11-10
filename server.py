import style_transfer_gpu as st
import pose_estimation_gpu as pe
import torchvision.transforms as transforms
import requests
import base64

from PIL import Image
from io import BytesIO
from flask import Flask, render_template, jsonify, request
from flask_restful import reqparse, abort, Resource, Api


app = Flask(__name__)
api = Api(app)

ENCODING = 'utf-8'

class StyleTransfer(Resource):

    def post(self):

        # style_img = f"D:/workspace/Python/2019Hackerton/data/neural/style/{request.form.get('style')}.jpg"
        # custom_style = request.form.get("style2")
        # print(custom_style)
        # if custom_style != None:
        #     style_img = self.get_image_from_url(custom_style)

        style_img = request.form.get("style")    
        content_url = request.form.get("content")
        content_img = self.get_image_from_url(content_url)
        
        print(style_img)

        transfer = st.StyleTransfer(style_img, content_img, 256)
        output = transfer.run().cpu().squeeze(0)
        im = transforms.ToPILImage()(output).convert("RGB")
        base64_string = base64.b64encode(self.image_to_bytes(im)).decode(ENCODING)

        data = { "status": "success", "img": content_url, "output": base64_string}
        return jsonify(data)

    def get_image_from_url(self, img_url):
        response = requests.get(img_url)
        image = BytesIO(response.content)
        return image

    def image_to_bytes(self, image):
        byteIO = BytesIO()
        image.save(byteIO, format='PNG')
        byteArr = byteIO.getvalue()
        return byteArr

class PoseEstimator(Resource):

    def post(self):
        # path = request.form.get('path')
        # max_num = request.form.get("maxnumber")

        # inputpath = [f"{path}{i+1}.jpg" for i in range(int(max_num))]
        inputpath = request.form.get('path')
        outputpath = "result/"
        
        estimator = pe.PoseEstimator(inputpath, outputpath)
        estimator.run()
        data = estimator.save_result()

        return jsonify(data)
        # return jsonify({"path": path, "max": max_num})

api.add_resource(StyleTransfer, '/style/')
api.add_resource(PoseEstimator, "/pose/")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
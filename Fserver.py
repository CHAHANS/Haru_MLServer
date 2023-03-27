#flask server
import sdModels

from flask import Flask, request
from flask import send_file, render_template, make_response

from PIL import Image
import base64

app = Flask(__name__)

template = '''
<html>
   <body>
      <form action = "http://116.38.220.14/result" method = "POST">
         <label for="doc">일기를 입력하세요</label>
         <input type ="text" id="doc" name="doc">
         <input type = "submit" value="Submit">
      </form>
   </body>
</html>
'''

def encode_image(image):
    with open(image, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    return encoded_string

@app.route('/')
def index(): 
    return template

@app.route('/result', methods=['POST'])
def result():
    sd_model = sdModels.sdModel()
    sd_model.getmodel(1)
    doc = request.form["doc"]
    sd_model.gettext(doc)
    img_path = "new.png"
    img = sd_model.modelrun()
   #  img.save(img_path)
    return send_file(img_path, mimetype='image/png')

# 원래 파일을 전달하던 방식
@app.route('/resultAPI/old', methods=['POST'])
def resultAPI_old():
    sd_model = sdModels.sdModel()
    sd_model.getmodel(1)
    json = request.get_json()
    doc = json["doc"]
    sd_model.gettext(doc)
    # img_path = "new.png"
    img = sd_model.modelrun()
    #img.save(img_path)
    return encode_image("new.png")

@app.route('/resultAPI', methods=['POST','OPTIONS'])
def resultAPI():
    if request.method == 'OPTIONS':
        response = make_response()
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST')
        return response

    json = request.get_json()
    print(json)
    sd_model = sdModels.sdModel()
    imgid = json["imageId"]
    sd_model.getmodel(json["modelId"])
    sd_model.getimageid(imgid)
    sd_model.gettext(json["doc"])
    
    # Tag
    # if json["tag"]:
    #     for 

    #모델실행
    sd_model.modelrun(imgid)
    url = f"http://116.38.220.14/imgview/{imgid}"
    return url

@app.route('/imgview/<int:id>')
def viewIMG(id):
    path = "/imges/"+str(id)+".png"
    return render_template("imgView.html", image_file=path)
    # 다운로드형식으로 보내고 싶다면
    # return send_file(path, mimetype='imgae/png', as_attachment=False)

host_addr="0.0.0.0"
port = "8080"

if __name__ == "__main__":
    app.run(host=host_addr, port=port, debug=True)
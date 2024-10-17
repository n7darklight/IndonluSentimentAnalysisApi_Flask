from flask import Flask, request
import json
import os
from flask_cors import CORS
from collections import OrderedDict

from sentiment import getSentimentAnalysis

script_dir = os.path.dirname(__file__)
UPLOAD_FOLDER = os.path.join(script_dir, 'files')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
CORS(app)

@app.route("/get_sentiment_analysis", methods=['GET','POST'])
def getSentiment():
   # query params
   content = request.json
   text = content['text']
   data = getSentimentAnalysis(text)
   print(data)

   result = {}
   if (data["status"] == 1):
        files = open(data["results"], "r")
        raw = json.loads(files.read(),object_pairs_hook=OrderedDict)
        #ba = customdecoder.decode(files.read())
        
        result = {
           "status": 200,
           "message": data["message"],
           "result": raw
       }
   else:
       result = {
           "status": 404,
           "message": data["message"],
           "result": {}
       }
   return result

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
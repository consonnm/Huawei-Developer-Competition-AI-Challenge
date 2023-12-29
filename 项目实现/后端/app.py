from flask import Flask, request, jsonify, json
import requests
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app, resources=r'/*')

@app.route('/queryResponse', methods=['POST'])
def queryResponse():
    prompt = request.form.get('prompt')
    print(prompt)
    history = request.form.get('history')
    url = 'https://iam.myhuaweicloud.com/v3/auth/tokens'
    payload = {
        "auth": {
            "identity": {
                "methods": ["password"],
                "password": {
                    "user": {
                        # Yes we changed our credentials before commiting, Stop wasting your time.
                        "domain": {"name": "blddzz"},
                        "name": "consonnm",
                        "password": "su123456,,"
                    }
                }
            },
            "scope": {
                "project": {"name": "cn-southwest-2"}
            }
        }
    }
    response = requests.post(url, json=payload)
    token = response.headers.get('X-Subject-Token')
    headers = {
        'x-auth-token': token,
    }
    data = {
        "prompt": prompt,
        "choices": ["A", "B", "C", "D"]
    }
    file_path = 'data.json'
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file,ensure_ascii=False)
    url = 'https://infer-modelarts-cn-southwest-2.myhuaweicloud.com/v1/infers/79022248-5ae4-47d6-9346-d171275fbfc8'
    with open(file_path, 'rb') as file:
        files = {'input_text': (file_path, file)}
        response = requests.post(url, files=files, headers=headers)
    txt =response.json()["result"]["response"]
    sep = "答："
    txt = txt.split(sep)[1] if sep in txt else ""
    re = {
        "response": txt,
        "history": response.json()["result"]["history"]
    }
    return jsonify(re)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

from flask import Flask

app = Flask(__name__)

# Hello Route
@app.route('/hello')
def hello():
    return "hello"

if __name__ == '__main__':
    app.run(debug=True)

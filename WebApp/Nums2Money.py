from flask import Flask, render_template, url_for, request
app = Flask(__name__)


@app.route('/')
@app.route('/Home')
@app.route('/Home', methods=['POST'])
def home():

    return render_template('home.html')

@app.route('/Info')
def info():
    return render_template('info.html', title='info')


if __name__ == '__main__':
    app.run(debug=True)

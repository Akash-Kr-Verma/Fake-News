from flask import Flask, request, render_template
from fake import predict_news  # Import your predict_news function

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        news_text = request.form['news_text']
        result = predict_news(news_text)  # Call your prediction function
    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
from flask import Flask, render_template, request, Response

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get the uploaded image file
        image = request.files['image']

        # Convert the image file to a base64 string
        import base64
        img_data = base64.b64encode(image.read()).decode('utf-8')

        # Pass the base64 string to the template
        return render_template('result.html', img_data=img_data)

    return render_template('result.html')

if __name__ == '__main__':
    app.run(debug=True)
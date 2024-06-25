from flask import Flask, request, jsonify
import util

app = Flask(__name__)

@app.before_first_request
def load_artifacts():
    util.load_artifacts()
    print("Artifacts loaded successfully.")

@app.route("/get_city_names")
def get_city_names():
    cities = util.get_city_names()
    print(f"City names retrieved: {cities}")
    response = jsonify({
        'city': cities
    })
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

@app.route("/predict_home_price", methods=['POST'])
def predict_home_price():
    try:
        # Debug statement to log request headers
        print("Request Headers:", request.headers)
        print("Request Data:", request.data)

        if request.content_type != 'application/json':
            return jsonify({'error': 'Bad Request', 'message': 'Content-Type must be application/json'}), 400

        data = request.get_json(force=True)  # Force parsing as JSON
        print("Parsed JSON data:", data)

        if not data:
            return jsonify({'error': 'Bad Request', 'message': 'Invalid JSON data'}), 400

        house_size = float(data['house_size'])
        city = data['city']
        bed = int(data['bed'])
        bath = int(data['bath'])

        estimated_price = util.get_estimated_price(city, house_size, bed, bath)

        response = jsonify({
            'estimated_price': estimated_price
        })
        return response

    except KeyError as e:
        return jsonify({'error': 'Bad Request', 'message': f'Missing key: {e.args[0]}'}), 400
    except ValueError as e:
        return jsonify({'error': 'Bad Request', 'message': f'Invalid value: {str(e)}'}), 400
    except Exception as e:
        # Log the exception for debugging
        print("Exception occurred:", str(e))
        return jsonify({'error': 'Internal Server Error', 'message': str(e)}), 500

if __name__ == "__main__":
    print("Starting server")
    util.load_artifacts()
    app.run(debug=True)

from flask import Flask, request, Response
from google.cloud import storage
from google.cloud import pubsub_v1

app = Flask(__name__)

# python http-client.py -d "127.0.0.1" -p "8080" -b "none" -w "files_get" -v -n 15 -i 10000
# python3 http-client.py -d "34.31.26.182" -p "8080" -b "none" -w "files_get" -v -n 5000 -i 10000
# python3 http-client.py -d "34.41.232.220" -p "8080" -b "none" -w "files_get" -v -n 5000 -i 10000

banned = ['north korea', 'iran', 'cuba', 'myanmar', 'iraq', 'libya', 'sudan', 'zimbabwe', 'syria']

def get_file_name():
    url_path = request.path
    # Split url path with '/' and get the last element
    file_name = url_path.split('/')[-1]
    return file_name

@app.route('/files_get/<filename>', methods=['GET','POST','PUT', 'DELETE', 'HEAD', 'CONNECT', 'OPTIONS', 'TRACE', 'PATCH'])
def files_get(filename):
    # Access the request method
    method = request.method

    country = None
    # Getting country
    if 'X-country' in request.headers:
        country = request.headers.get("X-country").lower().strip()
        print("Country: ", country)
    else:
        print("No Country")

    # Checking country and handling file retrieval
    if method != 'GET':
        print("Method not implemented: ", method)
        return 'Not Implemented', 501
    else:
        if country not in banned:
            print("Permission Granted")

            # Getting file contents
            file_name = filename  # Use the dynamic filename
            client = storage.Client()
            bucket = client.bucket('bu-ds561-jawicamp')
            blob = bucket.blob(file_name)

            try:
                # Returning file contents and OK status
                content = blob.download_as_text()
                response = Response(content, status=200, headers={'Content-Type': 'text/html'})

                return response

            except Exception as e:
                print("File not found: ", file_name)
                return str(e), 404
        else:
            print("Permission Denied")
            publisher = pubsub_v1.PublisherClient()
            topic_path = publisher.topic_path('jacks-project-398813', 'banned_countries')

            # Bytestring data
            data_str = f"{country}"
            data = data_str.encode("utf-8")

            # Try to publish
            try:
                future = publisher.publish(topic_path, data)
                future.result()  # Wait for the publish operation to complete
                print("Published to Pub/Sub successfully")
            except Exception as e:
                print("Error publishing to Pub/Sub:", str(e))
                return "Publish Denied", 400
            return "Permission Denied", 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
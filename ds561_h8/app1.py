from flask import Flask, request, Response
from google.cloud import storage
from google.cloud import pubsub_v1
import socket

app = Flask(__name__)

# python http-client.py -d "127.0.0.1" -p "8080" -b "none" -w "files_get" -v -n 15 -i 10000
# python3 http-client.py -d "34.31.26.182" -p "8080" -b "none" -w "files_get" -v -n 5000 -i 10000
# python3 http-client.py -d "35.188.147.248" -p "8080" -b "none" -w "files_get" -v -n 5000 -i 10000
# python3 http-client.py -d 35.188.147.248  -p 8080 -b "none" -w files_get -v -n 200 -i 9999 -f

banned = ['north korea', 'iran', 'cuba', 'myanmar', 'iraq', 'libya', 'sudan', 'zimbabwe', 'syria']

def get_filename():
    url_path = request.path
    filename = url_path.split('/')[-1]
    return filename

@app.route('/health', methods=['GET'])
def health_check():
    vm_name = socket.gethostname()  # Get the VM name
    return Response(f"OK: {vm_name}", status=200, headers={'Content-Type': 'text/plain'})

@app.route('/files_get/<filename>', methods=['GET','POST','PUT', 'DELETE', 'HEAD', 'CONNECT', 'OPTIONS', 'TRACE', 'PATCH'])
def files_get(filename):
    method = request.method

    country = None
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
            client = storage.Client()
            bucket = client.bucket('bu-ds561-jawicamp')
            blob = bucket.blob(filename)

            try:
                # Returning file contents and OK status
                content = blob.download_as_text()
                vm_name = socket.gethostname()  # Get the VM name
                content_with_vm = f"VM Name: {vm_name}\n{content}"
                response = Response(content_with_vm, status=200, headers={'Content-Type': 'text/html'})
                return response

            except Exception as e:
                print("File not found: ", filename)
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

# gcloud compute forwarding-rules describe www-rule --region us-central1
# while true; do curl -m1 35.188.147.248; done
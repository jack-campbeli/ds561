import functions_framework
from flask import Request, Response
from google.cloud import storage
from google.cloud import pubsub_v1

banned = ['north korea', 'iran', 'cuba', 'myanmar', 'iraq', 'libya', 'sudan', 'zimbabwe', 'syria']

def get_file_name(request: Request):
    url_path = request.path
    # split url path with '/' and get the last element
    file_name = url_path.split('/')[-1]
    
    return file_name

# https://us-central1-jacks-project-398813.cloudfunctions.net/files_get
@functions_framework.http
def files_get(request):
    """HTTP Cloud Function.
    Args:
       
    Returns:
       
    """
    method = request.method
    if method != 'GET':
        print("Method not implemented: ", method)
        return 'Not Implemented', 501
    else:
        country = None
        # getting country 
        if 'X-country' in request.headers:
            country = request.headers.get("X-country").lower().strip()
            print("Country: ", country)
        else:
            print("No Country")

        # checking country
        if country not in banned: # country is None or
            print("Permission Granted")
            
            # getting file contents
            file_name = get_file_name(request)
            client = storage.Client()
            bucket = client.bucket('bu-ds561-jawicamp')            
            blob = bucket.blob(file_name)

            try:
                # returning file contents and OK status
                content = blob.download_as_text()
                response = Response(content, status=200, headers={'Content-Type': 'text/html'})

                return response

            except Exception as e:
                print("File not found: ",file_name)
                return str(e), 404  
            
        else:
            print("Permission Denied")
            publisher = pubsub_v1.PublisherClient()
            topic_path = publisher.topic_path('jacks-project-398813', 'banned_countries')
    
            # bytestring data
            data_str = f"{country}"
            data = data_str.encode("utf-8")
            
            # try to publish
            try:
                future = publisher.publish(topic_path, data)
                future.result()  # Wait for the publish operation to complete
                print("Published to Pub/Sub successfully")
            except Exception as e:
                print("Error publishing to Pub/Sub:", str(e))
                return "Publish Denied", 400
            return "Permission Denied", 400
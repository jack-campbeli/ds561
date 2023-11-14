import re
import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions

from apache_beam.options.pipeline_options import GoogleCloudOptions
from apache_beam.options.pipeline_options import SetupOptions
from apache_beam.options.pipeline_options import StandardOptions


# bucket_name = 'bu-ds561-jawicamp'
# project_id = 'jacks-project-398813'

def parse_file(file):
    output = []
    filename, content = file
    links = re.findall(r'<a\s+[^>]*\bHREF=["\']([^"\']+\.html)["\'][^>]*>', content)
    
    for link in links:
        output.append((link, filename))
    return output

options = PipelineOptions()
google_cloud_options = options.view_as(GoogleCloudOptions)
google_cloud_options.project = 'jacks-project-398813'
google_cloud_options.job_name = 'jawicamp'
google_cloud_options.region = 'us-central1'
google_cloud_options.staging_location = 'gs://bu-ds561-jawicamp/staging'
google_cloud_options.temp_location = 'gs://bu-ds561-jawicamp/tmp'
options.view_as(SetupOptions).save_main_session = True
options.view_as(StandardOptions).runner = 'DataflowRunner'

pipeline = beam.Pipeline(options=options)

links = (
        pipeline 
        | 'ReadFiles' >> beam.io.ReadFromTextWithFilename('gs://bu-ds561-jawicamp/*.html')
        | 'ParseFiles' >> beam.FlatMap(parse_file)
        )
# links | 'PrintLinks' >> beam.Map(lambda x: print(f"Processing: {x}"))

outgoing_counts = (
    links 
    | 'ExtractOutgoingLinks' >> beam.Map(lambda link_tuple: link_tuple[0])
    | 'CountOutgoingLinks' >> beam.Map(lambda link: (link, 1))
    | 'GroupAndSumOutgoing' >> beam.CombinePerKey(sum)
)
# outgoing_counts | 'LogOutgoingCounting' >> beam.Map(lambda x: print(f"Counting outgoing link: {x}"))

incoming_counts = (
    links 
    | 'ExtractIncomingLinks' >> beam.Map(lambda link_tuple: (link_tuple[0], link_tuple[1]))
    | 'MapIncomingLinks' >> beam.Map(lambda link_tuple: (link_tuple[0], link_tuple[1]))
    | 'GroupIncomingLinks' >> beam.GroupByKey()
    | 'CountUniqueIncomingLinks' >> beam.Map(lambda x: (x[0], len(set(x[1]))))
)

# results
top_incoming = incoming_counts | 'TopIncoming' >> beam.combiners.Top.Of(5, key=lambda x: x[1])
top_incoming | 'PrintTopIncoming' >> beam.Map(lambda x: print(f"Top Files (Incoming): {x}"))

top_outgoing = outgoing_counts | 'TopOutgoing' >> beam.combiners.Top.Of(5, key=lambda x: x[1])
top_outgoing | 'PrintTopOutgoing' >> beam.Map(lambda x: print(f"Top Files (Outgoing): {x}"))

output_file_path = "gs://bu-ds561-jawicamp/results"
top_incoming | 'WriteTopIncoming' >> beam.io.WriteToText(output_file_path + '/top_incoming')
top_outgoing | 'WriteTopOutgoing' >> beam.io.WriteToText(output_file_path + '/top_outgoing')

result = pipeline.run()
result.wait_until_finish()

##### ignore below #####

# for i in dataflow compute_component logging storage_component storage_api bigquery pubsub datastore.googleapis.com cloudresourcemanager.googleapis.com; do gcloud services enable $i; done

# for role in roles/dataflow.admin roles/dataflow.worker roles/storage.objectAdmin; do
#   gcloud projects add-iam-policy-binding jacks-project-398813 --member="serviceAccount:857377565329-compute@developer.gserviceaccount.com" --role="$role"
# done

# python processing.py \
#     --region central1-a \
#     --input gs://bu-ds561-jawicamp/ \
#     --output gs://bu-ds561-jawicamp/results/outputs \
#     --runner DataflowRunner \
#     --project jacks-project-398813 \
#     --temp_location gs://bu-ds561-jawicamp/tmp/

# gsutil ls -lh "gs://bu-ds561-jawicamp/results/outputs*"
# gsutil cat "gs://bu-ds561-jawicamp/results/outputs*"
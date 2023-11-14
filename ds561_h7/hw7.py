import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
from bs4 import BeautifulSoup

options = PipelineOptions()
p = beam.Pipeline(options=options)
files = p | 'ReadFiles' >> beam.io.ReadFromTextWithFilename('gs://bu-ds561-xyz/def/44.html')

def process(file_tuple):
    filename, content = file_tuple
    soup = BeautifulSoup(content, 'html.parser')
    links = [(a['href'], filename) for a in soup.find_all('a', href=True)]
    return links

# Process HTML content and extract links with their source file
links_with_source = files | 'ProcessHTML' >> beam.FlatMap(process)

# Separate outgoing and incoming link information
outgoing_links = links_with_source | 'ExtractOutgoingLinks' >> beam.Map(lambda link_tuple: link_tuple[0])
incoming_links = links_with_source | 'ExtractIncomingLinks' >> beam.Map(lambda link_tuple: (link_tuple[0], link_tuple[1]))

# Count outgoing links
outgoing_counts = (
    outgoing_links
    | 'CountOutgoingLinks' >> beam.Map(lambda link: (link, 1))
    | 'GroupOutgoingLinks' >> beam.GroupByKey()
    | 'CountOutgoingLinksSum' >> beam.Map(lambda x: (x[0], sum(x[1])))
)
# outgoing_counts | 'LogOutgoingCounting' >> beam.Map(lambda x: print(f"Counting outgoing link: {x}"))
# Count incoming links
incoming_counts = (
    incoming_links
    | 'MapIncomingLinks' >> beam.Map(lambda link_tuple: (link_tuple[0], link_tuple[1]))
    | 'GroupIncomingLinks' >> beam.GroupByKey()
    | 'CountUniqueIncomingLinks' >> beam.Map(lambda x: (x[0], len(set(x[1]))))
)
#incoming_counts | 'LogIncomingCounting' >> beam.Map(lambda x: print(f"Counting incoming link: {x}"))

# Identify top 5 outgoing links
top_outgoing = outgoing_counts | 'TopOutgoing' >> beam.combiners.Top.Of(5, key=lambda x: x[1])

# Log the top outgoing links
top_outgoing | 'LogTopOutgoing' >> beam.Map(lambda x: print(f"Top Outgoing Link: {x}"))

# Identify top 5 incoming links
top_incoming = incoming_counts | 'TopIncoming' >> beam.combiners.Top.Of(5, key=lambda x: x[1])

# Log the top incoming links
top_incoming | 'LogTopIncoming' >> beam.Map(lambda x: print(f"Top Incoming Link: {x}"))

# Run the pipeline
result = p.run()

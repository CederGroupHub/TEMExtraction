from pipeline_single import run_pipeline_single

f = open('sample_html.html', 'r')
data = str(f.read())
f.close()

run_pipeline_single('extracted_data_single', False, "rsc", data)


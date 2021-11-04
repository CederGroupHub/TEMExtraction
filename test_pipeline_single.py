from pipeline_single import run_pipeline_single

f = open('sample_html3.html', 'r')
data = str(f.read())
#print(data)
f.close()

run_pipeline_single('extracted_data_single', False, "rsc", data)


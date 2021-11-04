import os

from ImageSoup.ImageSoup.factory import factory, key_to_publisher

'''
Elsevier:
Failed for the following:
10.1016/j.ssc.2010.05.009
10.1016/j.optcom.2015.07.032
10.1016/j.bioelechem.2009.06.002
TODO(aksub99): investigate reasons for failures.

Nature:
TODO(aksub99): Fix nature nanotechnology (dois are like 10.1038/nnano.2015.33)

Springer:
Failure DOIs:
10.1007/s11244-018-0920-7
10.1007/s10562-018-2389-1
10.1007/s10562-014-1255-z
10.1007/s10562-017-2231-1
10.1007/s10800-013-0589-3
10.1007/s11244-013-0154-7
10.1007/s10562-017-2245-8
TODO(aksub99): Investigate reasons for failures.
'''

def extract_figures_single_paper(publisher: str, content: str, **kwargs):
    soup = factory.create(publisher)
    meta = {}
    if publisher == 'nature':
        if 'doi' not in kwargs or 'year' not in kwargs:
            raise(Exception('doi and year are not specified!')) 
        figures = soup.parse(content, doi=kwargs['DOI'], year=kwargs['year'])
    else:
        figures = soup.parse(content)
    # try:
    #     figures = soup.parse(content, doi=doc["DOI"])
    # except:
    #     print("No figures in paper!")
    
    meta["Publisher"] = publisher
    meta["Figures"] = figures["figures"]
    return meta
 



__all__= "morris"


from urllib import request



def read_url(url):
    with request.urlopen(url) as req:
        data = req.read()
    return data

def save_url(url, file):
    with request.urlopen(url) as req:
        with open(file, 'wb') as fid:
            while True:
                data = req.read(100000)
                if not data:
                    return
                fid.write(data)

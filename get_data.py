import requests

#download the csv files
url = "https://zenodo.org/record/2547147/files/eeg#.edf?download=1"
for i in range(1,80):
    this_url = url.replace("#", str(i))
    print(this_url)
    r = requests.get(this_url)
    file = open("./eeg-data/download/eeg"+str(i)+".edf", 'wb').write(r.content)
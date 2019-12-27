#!/usr/bin/env python3
import requests, json

baseurl = 'some_url_with_rest_api/GetDetails'

def downloadRaw(buildingId):
  # 1 is get from GetProjectsByArea
  # 22 is get from GetHouseEstatesByProject
  payload = {'ProjectId':'1', 'HouseEstateId':'22', 'BuildingId':buildingId}
  r = requests.post(baseurl, data=payload)
  print (r.request.body)
  res = r.json()
#  print res
  return res
  

def getTime():
  import time, datetime
  ts = time.time()
  return datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%H%M')

def saveToFile(buildingId):
  timePrefix = getTime()
  dirName = 'rawData'
  filename = dirName + '/{0}_{1}.json'.format(timePrefix, buildingId)

  import os.path
  if not os.path.exists(dirName):
    os.makedirs(dirName)
  if os.path.isfile(filename):
    print('File exists, skip download. ' + filename)
    return 0

  j = downloadRaw(buildingId)
  import io
  with io.open(filename, 'w', encoding='utf-8') as outfile:
#https://stackoverflow.com/questions/18337407/saving-utf-8-texts-in-json-dumps-as-utf8-not-as-u-escape-sequence
#    data = json.dumps(j, ensure_ascii=False)
#    outfile.write(unicode(data))
    json.dump(j, outfile, ensure_ascii=False)


def downloadAll():
  import time
  import random
  # make sure building starts with 3
  # Python3 must use list to convert to array
  buildings = list(range(1, 6))
  while buildings[0] != 3:
    random.shuffle(buildings)

  print(buildings)

  for b in buildings:
    print("Download for building {0} ...".format(b))
    saveToFile(b)
    time.sleep(2)


if __name__ == "__main__":
  downloadAll()

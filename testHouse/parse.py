#!/usr/bin/env python3
# In order to work with unicode string, using python 3!!!
import json, os

headers = []
items = []

for filename in os.listdir('rawData'):
  if filename[-5:] != '.json':
    continue

  print("Loading {0}...", filename)
  with open('rawData/'+filename, 'r') as json_file:
#    decoded_json = json_file.read()
    a = json.load(json_file)
    # Get current building number and all units
    bId = a["Building"]
    bUnits = len(a["Houses"])

    # Loop over each unit in current building
    for curUnit in a["Houses"]:
      unitNumber = curUnit["Unit"]
      # Loop over each house in current unit & building
      for house in curUnit["HouseDetail"]:
        if not headers:
          headers = list(house.keys())
          headers.sort()
          headers = ["WholeId"] + headers
        house["WholeId"] = "{}#{}-{}".format(bId, house["Unit"], house["RoomNumber"])

        items.append(house)
#    print()
    # Get current building number and all units

#    print(a["Houses"][0]["HouseDetail"][0])

print (headers)

# All get collected, now save to csv.
import csv
with open('summary.csv', 'w') as csvfile:
  csvWriter = csv.writer(csvfile)
  csvWriter.writerow(headers)
  for item in items:
    csvWriter.writerow([item[h] for h in headers])

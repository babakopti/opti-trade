import os, sys
import json

prtMod = '/var/prt_weights/'
out_file = 'portfolios/actual_wt_hash_Dec8_2020.json'

wt_hash = {}
for item in os.listdir(prtMod):
    filePath = os.path.join(prtMod, item)
    tmp = item.split('.')[0].split('_')
    date_str = '%s %s' % (tmp[2], tmp[3])
    print(date_str)
    tmp_hash = json.load(open(filePath, 'r'))
    wt_hash[date_str] = tmp_hash

json.dump(wt_hash, open(out_file, 'w'))



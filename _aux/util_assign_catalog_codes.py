import json

txt_file = "catalogue_codes.txt"
jsn_file = "astCat_photCat.json"
out_file = "AstCatWithCodes.json"


with open(jsn_file, "r") as f:
    raw_json = json.load(f)


for entry in raw_json:
    entry['OldValue'] = entry['Value']

for entry in raw_json:
    if entry['Value'].startswith('PS1_DR'):
        entry['Value'] = entry['Value'].replace('_', '')
    if entry['Value'] == 'Gaia3':
        entry['Value'] = 'GaiaDR3'
    if entry['Value'] == 'Gaia3E':
        entry['Value'] = 'GaiaEDR3'
    if entry['Value'] == 'Gaia1':
        entry['Value'] = 'GaiaDR1'
    if entry['Value'] == 'Gaia2':
        entry['Value'] = 'GaiaDR2'
    if entry['Value'] == 'Hip1':
        entry['Value'] = 'Hipparcos'
    if entry['Value'] == 'Hip2':
        entry['Value'] = 'Hipparcos 2'
    if entry['Value'] == 'Tyc1':
        entry['Value'] = 'Tycho1'
    if entry['Value'] == 'Tyc2':
        entry['Value'] = 'Tycho2'
    if entry['Value'] == 'UBSC':
        entry['Value'] = 'USNOUBAD'
    if entry['Value'] == 'GSC':
        entry['Value'] = 'GSC (version unspecified)'
    if entry['Value'] == 'SDSS7':
        entry['Value'] = 'SDSSDR7'
    if entry['Value'] == 'SDSS8':
        entry['Value'] = 'SDSSDR8'    
    if entry['Value'] == 'SAO1984':
        entry['Value'] = 'SAO 1984'
    if entry['Value'] == 'AGK3':
        entry['Value'] = 'AGK 3'
    if entry['Value'] == 'LickGas':
        entry['Value'] = 'Lick Gaspra Catalogue'
    if entry['Value'] == 'Ida93':
        entry['Value'] = 'Ida93 Catalogue'
    if entry['Value'] == 'Perth70':
        entry['Value'] = 'Perth 70'
    if entry['Value'] == 'COSMOS':
        entry['Value'] = 'COSMOS/UKST Southern Sky Catalogue'


cat_codes = {}
with open(txt_file, "r") as f:
    next(f) 
    count = 0
    for line in f:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        #code, name = line.split()[0], line.split()[1]
        #print(count, code, name) 
        parts = line.split(maxsplit=1)
        code = parts[0]
        name = parts[1]
        name = name.replace('-', '')
        if name.startswith('USNO'):
           name = name.replace('.0', '')
        cat_codes[name] = code
        count += 1



count_unmatched = 0
for entry in raw_json:
    if entry['Value'] in cat_codes:
        entry['Code'] = cat_codes[entry['Value']]
    else:
        entry['Code'] = ' '
        print(entry['Value'])
        count_unmatched += 1




key_name = 'OldValue'
placeholder = 'Unknown'
updated_json = {}
for entry in raw_json:
    k = entry[key_name]
    # copy all other fields
    values = {field: entry.get(field, placeholder) 
              for field in entry if field != key_name}
    updated_json[k] = values


with open(out_file, 'w') as f:
    json.dump(updated_json, f, indent=2)





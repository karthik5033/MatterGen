
import json
import zipfile
import pprint

ZIP_PATH = r"d:\coding_files\Projects\matterGen\material dataset\mp_20.json.zip"

with zipfile.ZipFile(ZIP_PATH, 'r') as z:
    for filename in z.namelist():
        if filename.endswith('.json'):
            with z.open(filename) as f:
                content = json.load(f)
                if isinstance(content, dict) and 'data' in content: content = content['data']
                
                entry = content[0]
                with open("entry_dump.txt", "w") as out:
                    out.write(pprint.pformat(entry))
                break

from tqdm import tqdm
from utils.coco_instance import *
import json

cocosem = COCOSemantic(sample_by_category=True)
for i in tqdm(range(len(cocosem)), total=len(cocosem)):
    cocosem.__getitem__(i, add_sample=True)
samples = cocosem.samples_by_category

json_file = "utils/samples_0.05_0.25.json"
with open(json_file, 'w') as f:
    json.dump(samples, f)
    print("Save samples to {}".format(json_file))
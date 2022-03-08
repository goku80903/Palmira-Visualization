import json
import numpy as np
import sys

"""
Assumes ground truth annotation, groundTruth label can be changed according to data.
"""

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

with open(sys.argv[1],"r") as f:
    data = json.load(f)

formatted_data = {}
images_root = "/home2/dama.sravani/Palmira/data"

with open('../train.json','r') as f:
    check_temp = json.load(f)
filename2id = {}
for i in check_temp['images']:
    filename2id[i['file_name']] = i['id']


img_dir = images_root
#json_file = os.path.join(sys.argv[1], "via_region_data.json")

for idx, v in enumerate(data["_via_img_metadata"].values()):
    temp = {}
    url = v["filename"]
    filename = url.replace("%20", " ")

    file_name1 = ""
    bhoomi = ["Bhoomi_data", "bhoomi"]
    #bhoomi = []
    if any(x in filename for x in bhoomi):
        #continue
        if "images" in filename:
            f = filename.split("images")[1]
            file_name1 = img_dir + "/Bhoomi_data/images" + f
        else:
            f = filename.split("bhoomi")[1]
            file_name1 = img_dir + "/Bhoomi_data/images" + f
    collections = ["penn_in_hand", "penn-in-hand", "jain-mscripts", "ASR_Images"]
    #collections = ["jain-mscripts"]
    if "penn_in_hand" in filename:
        file_name1 = img_dir + filename.split("9006")[1]
        #continue
    if "penn-in-hand" in filename:
        #continue
        file_name1 = img_dir + filename.split("imgdata")[1]
    if "ASR_Images" in filename:
        #continue
        file_name1 = img_dir + filename.split("imgdata")[1]
    if "jain-mscripts" in filename:
        #continue
        file_name1 = img_dir + filename.split("imgdata")[1]
    #print(file_name1)
    temp["imagePath"] = file_name1
    temp["regions"] = []

    try:
      currID = filename2id[file_name1.replace("/home2/dama.sravani/Palmira/data", '/content/drive/MyDrive/images_')]
    except: 
      print(file_name1)
      continue

    annos = v["regions"]
    for ins in annos :
        shape = ins["shape_attributes"]
        temp2 = {}
        temp2["groundTruth"] = np.stack( (np.array(shape["all_points_x"]), np.array(shape["all_points_y"])), axis=1 )
        labelName = ins["region_attributes"]["Spatial Annotation"]
        if type(labelName) is list:
            labelName = labelName[0]
        if labelName == "Decorator":
            labelName = "Picture / Decorator"
        if labelName == "Picture":
            labelName = "Picture / Decorator"
        temp2['regionLabel'] = labelName
        temp2['id'] = str(currID)
        temp['regions'].append(temp2)

    formatted_data[str(currID)] = temp


with open("/".join(sys.argv[1].split("/")[:-1])+"/"+(sys.argv[1].split("/")[-1]).split(".")[0]+"-coco.json","w") as f:
    json.dump(formatted_data,f,cls=NumpyEncoder)

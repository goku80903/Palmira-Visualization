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
img_dir = images_root
#json_file = os.path.join(sys.argv[1], "via_region_data.json")

for region in data["_via_img_metadata"]:

    if region["image_id"] not in formatted_data.keys():
    
        temp = {}
        url = region["filename"]
        filename = url.replace("%20", " ")

        bhoomi = ["Bhoomi_data", "bhoomi"]
        #bhoomi = []
        file_name1 = ""
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
                

        #temp["imagePath"] = next((item for item in data["images"] if item["id"] == region["image_id"]),None)["file_name"]

        #temp2 = {}
        #temp["imagePath"] = temp["imagePath"].replace("/content/drive/MyDrive/images_", "/home2/dama.sravani/Palmira/data")

        #temp2["groundTruth"] = np.stack((np.array(region["segmentation"][0][::2]),np.array(region["segmentation"][0][1::2])),axis=1)
        #temp2["regionLabel"] = next((item for item in data["categories"] if item["id"] == region["category_id"]), None)["name"]
        #temp2["id"] = str(region["image_id"])
        #temp["regions"] = [temp2]
        annos = region["regions"]
        for ins in anno :
            shape = anno["shape_attributes"]
            temp2 = {}
            temp2["groundTruth"] = np.stack( (np.array(shape["all_points_x"]), np.array(shape["all_points_y"])), axis=1 )
        formatted_data[region["image_id"]] = temp
    
    else:
        temp2 = {}
        
        # only segmentation of list type supported for now

        if isinstance(region["segmentation"],list):
            temp2["groundTruth"] = np.stack((np.array(region["segmentation"][0][::2]),np.array(region["segmentation"][0][1::2])),axis=1)
        else:
            continue

        temp2["regionLabel"] = next((item for item in data["categories"] if item["id"] == region["category_id"]), None)["name"]
        temp2["id"] = str(region["image_id"])
        temp["regions"] = [temp2]

        formatted_data[region["image_id"]]["regions"].append(temp2)


with open("/".join(sys.argv[1].split("/")[:-1])+"/"+(sys.argv[1].split("/")[-1]).split(".")[0]+"-coco.json","w") as f:
    json.dump(formatted_data,f,cls=NumpyEncoder)


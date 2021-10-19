import json
import os


def get_list(path):
    pt_list = []
    for _, dirs, _ in os.walk(path):
        for dirname in sorted(dirs):
            for _, _, files in os.walk(os.path.join(path, dirname)):
                for filename in files:
                    if 'depth2pcd_GT' in filename:
                        name, extension = os.path.splitext(filename)
                        obj_idx = name[-1]  # the last char is the object index
                        pt_list.append("%s-%s" % (dirname, obj_idx))
    return pt_list


train_list = get_list("frankascanv2/train")
test_list = get_list("./frankascanv2/test")

frankascan = [{
    "taxonomy_id": "frankascanv2",
    "taxonomy_name": "FrankaScan-Beaker",
    "test": test_list,
    "train": train_list,
    "val": []
}]

if __name__ == '__main__':
    with open('FrankaScanv2.json', 'w') as outfile:
        json.dump(frankascan, outfile)

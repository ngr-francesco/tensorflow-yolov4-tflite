#!/bin/bash

### Recommended to run 'nohup ./<this_script> &' to prevent interruption from SSH session termination.

# Update for default OS specific package manager.
# sudo yum -y install java-1.8.0
# sudo yum -y remove java-1.7.0-openjdk

mkdir -p ../data/dataset/coco/images/ ../data/dataset/coco/annotations/

### 2017 COCO Dataset ###

echo "Downloading COCO dataset..."
curl -OL "http://images.cocodataset.org/zips/val2017.zip"  
curl -OL "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"

echo "Extracting..."

unzip 'val2017.zip' -d ../data/dataset/coco/images/
unzip 'annotations_trainval2017.zip' -d ../data/dataset/coco/ # Inflates to '/annotations'.

if $? -ne 0:
    echo "Unzipping wasn't successful. Aborting..."
    return 1

echo "Cleaning up zip files"

rm 'val2017.zip' 'annotations_trainval2017.zip'

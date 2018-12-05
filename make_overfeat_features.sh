#!/bin/bash
echo "starting"
for i in {00001..27638}; do convert Yummly28K/images27638/img"$i".jpg -resize 348x232! size232images/"$i".jpg;
./../../../Downloads/overfeat/bin/linux_64/overfeat -f size232images/"$i".jpg > overfeatures/features$i.out; echo "$i"; done
#!/bin/bash

sourcePath="/media/roderickmajoor/TREKSTOR/Train2/images/"
destinationPath="/media/roderickmajoor/TREKSTOR/Train2/images_split/"
partSize=3
count=0
folderCount=0

mkdir -p "$destinationPath"

for file in "${sourcePath}"*.jpg; do
    count=$((count+1))
    if [ $count -eq 1 ]; then
        folderCount=$((folderCount+1))
        mkdir -p "${destinationPath}${folderCount}/page"
    fi
    cp "$file" "${destinationPath}${folderCount}/"
    if [ $count -eq $partSize ]; then
        count=0
    fi
done

echo "Splitting and creating 'page' folders complete."


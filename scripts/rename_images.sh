#!/bin/bash

prefix="1045_"
sourcePath="/media/roderickmajoor/TREKSTOR/Train2/1045/images/"
sourcePattern="download-part0001-"
extension=".jpg"

cd "$sourcePath"

for file in ${sourcePattern}*${extension}; do
    number=$(echo $file | sed "s/${sourcePattern}//" | sed "s/${extension}//")
    newName="${prefix}${number}${extension}"
    mv "$file" "$newName"
done

echo "Renaming complete."


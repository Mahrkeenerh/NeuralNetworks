# Doesn't help much
find . -type f -iregex '.*\.hpp' -print0 | while read -d $'\0' file
do
    g++ -c "$file" -o "$file.gch"
done

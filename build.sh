time_start=$(date +%s%3N)

"g++" -fdiagnostics-color=always -g $(find . -type f -iregex '.*\.[c|h]pp') -Wall -fopenmp -Ofast -o main

time_end=$(date +%s%3N)
time_diff=$((time_end - time_start))
echo "Build time: $time_diff ms"

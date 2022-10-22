"g++" -fdiagnostics-color=always -g $(find . -type f -iregex '.*\.[c|h]pp') -fopenmp -Ofast -o main

"g++.exe" -fdiagnostics-color=always -g $(find . -type f -iregex '.*\.[c|h]pp') -o 'main' -fopenmp -Ofast

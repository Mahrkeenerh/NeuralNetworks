"g++.exe" -fdiagnostics-color=always -g $(find . -type f -iregex '.*\.cpp') -o 'main.exe' -fopenmp -Ofast

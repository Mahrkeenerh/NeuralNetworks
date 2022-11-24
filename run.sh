#!/bin/bash
module add gcc
dos2unix build.sh
bash build.sh
nice -n 19 ./main
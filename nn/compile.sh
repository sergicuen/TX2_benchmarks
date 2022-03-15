#!/bin/bash

for i in `seq 1 4444`;
do
    make clean_temp
    make arg=$i
done

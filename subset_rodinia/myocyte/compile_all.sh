#!/bin/bash

for i in `seq 1 10`;
do
    make clean_temp
    make arg=$i
done

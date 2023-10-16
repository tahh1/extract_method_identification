#!/bin/bash

# Add the project's root folder absoluite path
cwd_folder='C:/Users/Taha Draoui/Desktop/GraphCodeBERT/extract-method-identification'
echo $cwd_folder
cd $cwd_folder
filename=$cwd_folder/input.csv
echo here

# Split the input file into files of 10000 rows
split -d -a 4 --additional-suffix=.csv -l 10000 $filename $cwd_folder/data/file_


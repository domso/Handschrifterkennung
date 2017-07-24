#!/bin/bash


if [ $# -ne 2 ];
then
	echo "Require 2 parameters";
	exit 0;
fi

for file in $(find tests/$1 -type f)
do
	echo $file >> $2;
	./Release/mc17-team1 $file >> $2;
done

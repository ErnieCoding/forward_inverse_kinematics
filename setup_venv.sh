#!/bin/bash

if [ -d venv ]; then
	echo 'Virtual environment already exists. Abort.'
	exit
fi

virtualenv venv
source venv/bin/activate
pip install numpy scipy

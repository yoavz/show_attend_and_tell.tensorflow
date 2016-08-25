#!/bin/bash

# install tensorflow
sh ./setup-tensorflow.sh

# install matplotlib requirements
sudo apt-get install libpng-dev libfreetype6-dev libxft-dev

# update pip
sudo pip install -U pip

# install python libraries
sudo -H pip install -r requirements.txt

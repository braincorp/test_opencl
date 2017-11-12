#!/bin/bash

# Robust way of locating script folder
# from http://stackoverflow.com/questions/59895/can-a-bash-script-tell-what-directory-its-stored-in
SOURCE=${BASH_SOURCE:-$0}

DIR="$( dirname "$SOURCE" )"
while [ -h "$SOURCE" ]
do 
  SOURCE="$(readlink "$SOURCE")"
  [[ $SOURCE != /* ]] && SOURCE="$DIR/$SOURCE"
  DIR="$( cd -P "$( dirname "$SOURCE"  )" && pwd )"
done
DIR="$( cd -P "$( dirname "$SOURCE" )" && pwd )"

source $DIR/catkin_ws/devel/setup.sh

pythonpathadd() {
    # add folder to PYTHONPATH only if it exists and is not already in PYTHONPATH
    if [ -d "$1" ] && [[ ":$PYTHONPATH:" != *":$1:"* ]]; then
        PYTHONPATH="${PYTHONPATH:+"$PYTHONPATH:"}$1"
    fi
}

pythonpathadd $DIR
pythonpathadd $DIR/catkin_ws/src


# If you can not compile opencv on mac, there is a workaround to use brew opencv
# brew tap homebrew/science
# brew install opencv
# cp /usr/local/lib/python2.7/site-packages/cv* venv/lib/python2.7/site-packages/

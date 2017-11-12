#!/bin/bash
set -e

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
WDIR="$( pwd )"

cd $DIR
if [ ! -f src/CMakeLists.txt ]; then
  catkin_init_workspace src
fi

catkin_make -DCMAKE_BUILD_TYPE=RelWithDebInfo

cd $WDIR

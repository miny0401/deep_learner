#! /bin/bash
# 设置PYTHONPATH
get_script_path() {
    SOURCE=${BASH_SOURCE[0]}
    BASE_SCRIPT_PATH=$( dirname "$SOURCE" )
    while [ -h "$SOURCE" ]
    do
        SOURCE=$(readlink "$SOURCE")
        [[ $SOURCE != /* ]] && SOURCE="$DIR/$SOURCE"
        DIR=$( cd -P "$( dirname "$SOURCE"  )" && pwd )
    done
    BASE_SCRIPT_PATH=$( cd -P "$( dirname "$SOURCE" )" && pwd )
}

get_script_path

export PYTHONPATH=$BASE_SCRIPT_PATH/learner/

#!/bin/bash
set -o errexit

function usage() {
    echo "usage: sh $0 {test_case_list_file}"
}

if [ $# -lt 1 ]
then
    usage
    exit 1
fi

listfile=$1
base_path=$(cd `dirname $0`/..; pwd)
test_case_path=${base_path}/tests
export PYTHONPATH=$base_path:$PYTHONPATH

# install the require package
cd ${base_path}
pip install -r requirements.txt

# run all case list in the {listfile}
cd -
for test_file in `cat $listfile | grep -v ^#`
do
    echo "run test case ${test_file}"
	python ${test_case_path}/${test_file}.py
done

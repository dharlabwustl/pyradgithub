#!/bin/bash
SESSION_ID=${1}
XNAT_USER=${2}
XNAT_PASS=${3}
XNAT_HOST=${4}
TYPE_OF_PROGRAM=${5}
echo ${TYPE_OF_PROGRAM}::TYPE_OF_PROGRAM
if [[ ${TYPE_OF_PROGRAM} == "APPLY_PYRADIOMICS" ]] ;
then
  echo ${TYPE_OF_PROGRAM}
/workspace/venv/bin/python /software/call_pyradiomics.py
#    /software/processing_before_segmentation.sh $SESSION_ID $XNAT_USER $XNAT_PASS $XNAT_HOST  ##/input /output
fi

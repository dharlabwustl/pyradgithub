#!/bin/bash
SESSION_ID=${1}
XNAT_USER=${2}
XNAT_PASS=${3}
TYPE_OF_PROGRAM=${4}
echo TYPE_OF_PROGRAM::${TYPE_OF_PROGRAM}
#'https://redcap.wustl.edu/redcap/api/' #
echo ${REDCAP_API}
#export REDCAP_API=${6}
#echo REDCAP_API::${REDCAP_API}
# The input string
input=$XNAT_HOST ##"one::two::three::four"
# Check if '::' is present
if echo "$input" | grep -q "+"; then
  # Set the delimiter
  IFS='+'

  # Read the split words into an array
  read -ra ADDR <<< "$input"
  export XNAT_HOST=${ADDR[0]}
  SUBTYPE_OF_PROGRAM=${ADDR[1]}
else
export XNAT_HOST=${5}
    echo "'+' is not present in the string"
fi


echo ${TYPE_OF_PROGRAM}::TYPE_OF_PROGRAM::${SUBTYPE_OF_PROGRAM}::${ADDR[0]}::${ADDR[2]}::${ADDR[3]}
if [[ ${TYPE_OF_PROGRAM} == 'APPLY_PYRADIOMICS' ]]; then
/workspace/venv/bin/python call_pyradiomics.py ${SESSION_ID} '/input' ${ADDR}


#  /software/lin_transform_before_deepreg_mni_template.sh $SESSION_ID $XNAT_USER $XNAT_PASS $XNAT_HOST /input /output
fi

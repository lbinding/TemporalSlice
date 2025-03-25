#!/bin/bash 
#
# TemporalSlice Inference Script 
#   Takes required input, passes to a python script (fools proof) 
#   For compliments please contact lawrence.binding.19(at)ucl.ac.uk
#   
#   Created: 10/01/2025

# Get script name
script_name=`basename $0`

# Get script location
main_dir="$(dirname "$(realpath "$BASH_SOURCE")")"

# Loop over arguments looking for -in (-i) and -out (-o)
args=("$@")
i=0
while [ $i -lt $# ]; do
    if ( [ ${args[i]} = "-in" ] || [ ${args[i]} = "-i" ]) ; then
      let i=$i+1
      input_file=${args[i]}
    elif ( [ ${args[i]} = "-out" ] || [ ${args[i]} = "-o" ]) ; then
      let i=$i+1
      output_file=${args[i]}
    fi
    let i=$i+1
done

# Check if user gave correct inputs
if [ -z "${input_file}" ] || [ -z "${output_file}" ]; then
    correct_input=0
else 
    correct_input=1
fi

#Check the user has provided the correct inputs
if ( [[ ${correct_input} -eq 0 ]] ) ; then
  echo ""
  echo "Incorrect input. Please see below for correct use"
  echo ""
  echo "Options:"
  echo " -in:         Input MRI (T1) image   -- REQUIRED"
  echo " -out:        Resection Mask Output  -- REQUIRED"
  echo ""
  echo "${script_name} -in T1_image.nii.gz -out T1_image_RM.nii.gz"
  exit
fi

python -m temporalSlice.inference --in ${input_file} --out ${output_file}


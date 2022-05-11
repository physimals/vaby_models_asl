# Example fitting of ASL model
#
# This example uses the sample multi-PLD data from the FSL course
import sys

import matplotlib.pyplot as plt
import nibabel as nib

from vaby.main import run

model = "aslrest"
outdir = "example_avb_out"

options = {
    "data" : "asldata_diff.nii.gz",
    "mask" : "asldata_mask.nii.gz",
    "model_name" : model,
    "output" : outdir,
    "method" : "svb",
    "tau" : 1.8,
    "casl" : True,
    "plds" : [0.25, 0.5, 0.75, 1.0, 1.25, 1.5],
    "repeats" : [8],
    "slicedt" : 0.0452,
    "inferart" : True,
    "param_overrides" : {
        "ftiss" : {
            "prior_type" : "M",
        }
    },
    "epochs" : 300,
    "learning_rate" : 0.1,
    "batch_size" : 6,
    "save_mean" : True,
    "save_noise" : True,
    #"save_param_history" : True,
    #"save_free_energy_history" : True,
    "save_runtime" : True,
    #"save_free_energy" : True,
    "save_model_fit" : True,
    "save_log" : True,
    "save_input_data" : True,
    "save_var" : True,
    "log_stream" : sys.stdout,
}

# Run fabber as a comparison if desired
#import os
#os.system("fabber_asl --model=aslrest --method=spatialvb --param-spatial-priors=MN+ --noise=white --data=asldata_diff --mask=asldata_mask --max-iterations=50 --incart --inferart --tau=1.8 --casl --batsd=1.0 --bat=1.3 --ti1=2.05 --ti2=2.3 --ti3=2.55 --ti4=2.8 --ti5=3.05 --ti6=3.3 --slicedt=0.0454 --inctiss --infertiss --repeats=8 --output=asl_example_fabber_out --incbat --inferbat --overwrite --save-model-fit")

runtime, avb = run(**options)

# Display a single slice (z=10)
ftiss_img = nib.load("%s/mean_ftiss.nii.gz" % outdir).get_fdata()
delttiss_img = nib.load("%s/mean_delttiss.nii.gz" % outdir).get_fdata()
plt.figure("F")
plt.imshow(ftiss_img[:, :, 10].squeeze())
plt.figure("delt")
plt.imshow(delttiss_img[:, :, 10].squeeze())
plt.show()

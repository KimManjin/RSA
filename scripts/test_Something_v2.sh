##################################################################### 
# Parameters!
mainFolder="net_runs"
subFolder=$1
ckptName=$2
snap_pref="resnet"
entityName="dyconv"
projName="Dynamic_conv_ssv2"
expName=${subFolder}
train_path="data/something_v2_train.txt"
val_path="data/something_v2_val.txt"
test_path="data/something_v2_val.txt"

expFolder=${mainFolder}/${subFolder}
logFolder=${mainFolder}/${subFolder}
ckptFolder=${mainFolder}/${subFolder}
wandb=""

#############################################
#--- RSA hyperparams ---
transform="RSA"
position="[[2],[1,3],[1,3,5],[1]]"
kernel_size="[5,7,7]"
nh=8
dk=0
dv=0
dd=0
kernel_type="VplusR"
feat_type="VplusR"

#--- training hyperparams ---
dataset_name="something"
netType="ResNet"
batch_size=64
learning_rate=0.02 # lr follows root of bs
num_segments=$3
mode=1
dropout=0.3
iter_size=1
num_workers=16
epochs=80
label_smoothness=0.1
stochastic_depth=0.2
mixup_alpha=0.0

####################################

mkdir -p ${expFolder}/training
mkdir -p ${expFolder}/validation

echo "Current network folder: "
echo ${expFolder}


lastCheckpoint=${ckptFolder}/${ckptName}
echo "Testing checkpoint: ${lastCheckpoint}"

####################################
if test -f "${lastCheckpoint}"; then

python -u main.py ${dataset_name} RGB ${train_path} ${val_path} ${test_path}\
    --entity_name ${entityName} --proj_name ${projName} --exp_name ${expName} \
    --arch ${netType} --mode ${mode} --consensus_type avg \
    -b ${batch_size} --num_segments ${num_segments} \
    -tfm ${transform} -pos ${position} -ks ${kernel_size} -nh ${nh} -dk ${dk} -dv ${dv} -dd ${dd} -ktype ${kernel_type} -ftype ${feat_type} \
    --lr ${learning_rate} --lr_steps 30 40 --epochs ${epochs} --nesterov "True"\
    --cosine_lr --warmup 5 \
    --loss_type 'smooth_nll' --label_smoothness ${label_smoothness} --gd 100 -i ${iter_size} -j ${num_workers} --dropout ${dropout} --no_partialbn --pretrained_parts finetune\
    --stochastic_depth ${stochastic_depth} --mixup_alpha ${mixup_alpha} \
    --eval-freq 1 -p 20 \
    --snapshot_pref ${expFolder}/${snap_pref} --rgb_prefix img_ \
    --log_dir ${logFolder} ${wandb} \
    --resume ${lastCheckpoint} \
    -e --val_output_folder ${expFolder}/validation \
    
else
     echo "No file exists"

fi
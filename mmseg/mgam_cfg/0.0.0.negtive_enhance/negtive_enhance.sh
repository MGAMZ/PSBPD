set +e
echo "实验启动：负样本增强实验"

exp_name="2.negtive_enhance"

for round in 1 2 3 4 5
do

echo "第${round}次实验"
exp_round="work_dirs/${exp_name}_${round}"
python tools/train.py configs/mgam/${exp_name}/ConvNext.py --work-dir "${exp_round}/ConvNext"
python tools/train.py configs/mgam/${exp_name}/MAE.py --work-dir "${exp_round}/MAE"
python tools/train.py configs/mgam/${exp_name}/Poolformer.py --work-dir "${exp_round}/Poolformer"
python tools/train.py configs/mgam/${exp_name}/Resnet50.py --work-dir "${exp_round}/Resnet50"
python tools/train.py configs/mgam/${exp_name}/Segformer.py --work-dir "${exp_round}/Segformer"
python tools/train.py configs/mgam/${exp_name}/SegNext.py --work-dir "${exp_round}/SegNext"
python tools/train.py configs/mgam/${exp_name}/SwinTransformerV2.py --work-dir "${exp_round}/SwinTransformerV2"

done

echo "实验结束：负样本增强实验"
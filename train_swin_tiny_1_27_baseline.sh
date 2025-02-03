# setup keys
export DETECTRON2_DATASETS=/input/jieyuz2/weikaih/data/mask2former_dataset/datasets
export DATASETS=/input/jieyuz2/weikaih/data/mask2former_dataset/datasets
export WANDB_API_KEY=f773908953fc7bea7008ae1cf3701284de1a0682


cd /input/jieyuz2/weikaih/improve_segment/Mask2Former/mask2former/modeling/pixel_decoder/ops
sh ./make.sh
cd /input/jieyuz2/weikaih/improve_segment/Mask2Former

# pip install numpy==1.24
# pip uninstall numpy
# pip install numpy
# check python
# conda deactivate
# export PATH="/opt/conda/bin:$PATH"
# echo '"Conda environment deactivated. Current Python version is: $(which python)"
# echo "Python version: $(python --version)"

# python synthetic_panoptic2detection_coco_format_polygon.py --things_only

timestamp=$(date +%Y%m%d_%H%M%S)


python train_net.py \
    --num-gpus 4 \
    --config-file configs/coco/instance-segmentation/maskformer2_R50_bs16_50ep.yaml \
    SOLVER.MAX_ITER 20000 \
    OUTPUT_DIR outputs/coco_baseline/baseline
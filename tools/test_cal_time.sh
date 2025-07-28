# 获取当前脚本所在目录
SCRIPT_DIR="$(pwd)"

# 删除所有包含 "oneformer3d" 的路径
CLEANED_PYTHONPATH=$(echo "$PYTHONPATH" | tr ':' '\n' | grep -v "oneformer3d" | tr '\n' ':')

# 重新设置 PYTHONPATH，仅包含当前目录
export PYTHONPATH="$SCRIPT_DIR:$CLEANED_PYTHONPATH"


CUDA_VISIBLE_DEVICES=1 python tools/test.py \
    configs/oneformer3d_dab_mean_mamba_voxel.py \
    work_dirs/best_all_ap_iter_55000.pth
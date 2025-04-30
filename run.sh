sudo fuser -v /dev/nvidia0 |awk '{for(i=1;i<=NF;i++)print "kill -9 " $i;}' | sudo sh

python -m torch.distributed.run --nproc-per-node=8 dpo_select_bind.py --select_type min --select_num 10000 --interval 100 \
    --embedding-root ../REPA-E/svd_feat_10k_min_interval_100 \
    --embedding-json ../REPA-E/svd_feat_10k_min_interval_100/total_json.json \
    --target-mean 0.0

python -m torch.distributed.run --nproc-per-node=8 dpo_select_dc.py --select_type middle --select_num 10000 --interval 4 \
    --embedding-root ../REPA-E/svd_feat_10k_middle_040 \
    --embedding-json ../REPA-E/svd_feat_10k_middle_040/total_json.json \
    --target-mean 0.40

python -m torch.distributed.run --nproc-per-node=8 dpo_select_dc.py --select_type middle --select_num 10000 --interval 4 \
    --embedding-root ../REPA-E/svd_feat_10k_middle_043 \
    --embedding-json ../REPA-E/svd_feat_10k_middle_043/total_json.json \
    --target-mean 0.43
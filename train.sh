python scripts/resnet18_train.py \
    --run-name resnet_first \
    --dataset-dir /content/drive/MyDrive/colab_data/animals \
    --learning-rate 0.001 \
    --batch-size 32 \
    --num-workers 2 \
    --num-epochs 50 \
    --load-checkpoints 0 \
    --load-checkpoints-path /home \
    --save-checkpoints 1 \
    --save-checkpoints-epoch 10 \
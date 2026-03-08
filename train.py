from ultralytics import YOLO

def main():
    """
    YOLOv11 训练脚本 - 包含优化参数配置
    支持：自适应学习率、早停、数据增强、混合精度训练等
    """
    # 加载预训练模型
    model = YOLO("yolo11n.pt")

    # ======================== 训练参数配置 ========================
    model.train(
        # -------- 数据集配置 --------
        data="data/medicine-box.v7i.yolov11/data.yaml",  # 数据集配置文件路径
        imgsz=640,                    # 输入图像尺寸

        # -------- 训练轮次与批次 --------
        epochs=100,                   # 训练轮次（建议 50-200）
        batch=16,                     # 批次大小（根据显存调整，-1 为自动）
        workers=4,                    # 数据加载线程数

        # -------- 学习率配置（自适应调整） --------
        lr0=0.01,                     # 初始学习率
        lrf=0.01,                     # 最终学习率 = lr0 * lrf（余弦退火终点）
        momentum=0.937,               # SGD 动量 / Adam beta1
        weight_decay=0.0005,          # 权重衰减，防止过拟合
        warmup_epochs=3.0,            # 学习率预热轮次
        warmup_momentum=0.8,          # 预热阶段初始动量
        warmup_bias_lr=0.1,           # 预热阶段偏置学习率

        # -------- 优化器选择 --------
        optimizer='auto',             # 优化器: auto, SGD, Adam, AdamW, NAdam, RAdam, RMSProp
        cos_lr=True,                  # 使用余弦退火学习率调度器（自适应调整学习率）

        # -------- 早停机制 --------
        patience=20,                  # 早停耐心值：验证集指标连续 N 轮无提升则停止

        # -------- 数据增强 --------
        augment=True,                 # 启用数据增强
        hsv_h=0.015,                  # 色调增强幅度
        hsv_s=0.7,                    # 饱和度增强幅度
        hsv_v=0.4,                    # 亮度增强幅度
        degrees=0.0,                  # 旋转角度范围 (-degrees, +degrees)
        translate=0.1,                # 平移比例
        scale=0.5,                    # 缩放比例
        shear=0.0,                    # 剪切角度
        perspective=0.0,              # 透视变换
        flipud=0.0,                   # 上下翻转概率
        fliplr=0.5,                   # 左右翻转概率
        mosaic=1.0,                   # Mosaic 增强概率
        mixup=0.0,                    # MixUp 增强概率
        copy_paste=0.0,               # 复制粘贴增强概率

        # -------- 训练加速 --------
        amp=True,                     # 混合精度训练（FP16），加速训练、减少显存
        cache=False,                  # 缓存图像到内存/磁盘，加速训练（True/'ram'/'disk'）

        # -------- 模型保存与日志 --------
        name='yolo11n-medicine-box',  # 训练名称，保存到 runs/detect/name
        project='runs/train',         # 项目保存路径
        exist_ok=False,               # 是否覆盖已有训练目录
        save=True,                    # 保存训练检查点
        save_period=-1,               # 每 N 轮保存一次（-1 仅保存最佳和最后）
        val=True,                     # 训练时进行验证
        plots=True,                   # 生成训练曲线图

        # -------- 其他高级配置 --------
        rect=False,                   # 矩形训练（减少填充，适合长宽比差异大的图像）
        resume=False,                 # 从上次中断处恢复训练
        freeze=None,                  # 冻结前 N 层（迁移学习时使用，如 freeze=10）
        dropout=0.0,                  # 分类头 Dropout 比例
        seed=0,                       # 随机种子，保证可复现性
        deterministic=True,           # 确定性训练模式
        single_cls=False,             # 将所有类别合并为单类训练
        verbose=True,                 # 详细输出
    )


if __name__ == '__main__':
    main()
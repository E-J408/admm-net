from generate_data import OFDMDatasetGenerator


if __name__ == '__main__':
    data_gen = OFDMDatasetGenerator(
        Nb=10,
        Nd=10,
        L_max=3,
        snr_range=(5, 20),
        data_dir='data/testDataGen5'
    )

    # 生成数据
    train_data, val_data, test_data = data_gen.generate_complete_dataset(
        total_samples=10000,
        train_ratio=0.7,
        val_ratio=0.2,
        test_ratio=0.1
    )

    data_gen.visualize_dataset_stats()

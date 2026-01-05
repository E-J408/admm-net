from generate_data import DatasetGeneratorCreatePhi

if __name__ == '__main__':
    data_gen = DatasetGeneratorCreatePhi(
        Nb=10,
        Nd=10,
        L_max=3,
        snr_range=(20, 20),
        data_dir='../data/phi_fixSNR20L3_5000_least5'
    )
    data_gen.generate_complete_dataset(
        total_samples=5000,
        train_ratio=0.7,
        val_ratio=0.2,
        test_ratio=0.1
    )
    data_gen.visualize_dataset_stats()

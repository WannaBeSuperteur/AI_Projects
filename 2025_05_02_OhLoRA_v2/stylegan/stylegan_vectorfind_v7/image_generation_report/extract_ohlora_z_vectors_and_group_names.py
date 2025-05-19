import pandas as pd

ohlora_z_idxs = [127, 672, 709, 931, 1017, 1073, 1162, 1211, 1277, 1351,
                 1359, 1409, 1591, 1646, 1782, 1788, 1819, 1836, 1905, 1918,
                 2054, 2089, 2100, 2111, 2137, 2185, 2240]


if __name__ == '__main__':
    test_result_csv = pd.read_csv('test_result.csv')
    ohlora_z_info_cols = ['case', 'vector_no', 'group_name', 'eyes_corr', 'mouth_corr', 'pose_corr']
    ohlora_z_group_names_df = test_result_csv.iloc[ohlora_z_idxs, :][ohlora_z_info_cols]
    ohlora_z_group_names_df.to_csv('../ohlora_z_group_names.csv', index=False)

    latent_code_csv = pd.read_csv('latent_codes_all.csv')
    ohlora_z_vectors_df = latent_code_csv.iloc[ohlora_z_idxs, :]
    ohlora_z_vectors_df.to_csv('../ohlora_z_vectors.csv', index=False)

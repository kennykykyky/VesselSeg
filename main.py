import glob
import pandas as pd
import os
import matplotlib.pyplot as plt
import pdb

name_exp2name_model = {
    '01_03_01': 'ours',
    # '03_00': 'ours',
    '02_00': 'U-Net', # unet
    # '02_01_te01': 'ResU-Net', # res unet
    '02_01': 'ResU-Net', # res unet
    '02_06': 'Attention U-Net', # attention unet
    '02_13': 'Attention ResU-Net', # attention res-unet
}

# plt.figure()
# for name_exp, name_model in name_exp2name_model.items():
#     path_exp = glob.glob(f'save/test_{name_exp}*')[0]
#     path_csv = os.path.join(path_exp, 'precision_recall_curve.csv')
#     df = pd.read_csv(path_csv)
#     plt.plot(df['threshold'][:-1], df['recall'][:-1], label=name_model)
# plt.xlabel('Threshold')
# plt.ylabel('Recall')
# plt.legend()
# plt.show()

# pdb.set_trace()

plt.figure()
for name_exp, name_model in name_exp2name_model.items():
    path_exp = glob.glob(f'save/test_{name_exp}*')[0]
    path_csv = os.path.join(path_exp, 'precision_recall_curve.csv')
    df = pd.read_csv(path_csv)
    plt.plot(df['recall'], df['precision'], label=name_model)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()
plt.show()
#
# plt.figure()
# for name_exp, name_model in name_exp2name_model.items():
#     path_exp = glob.glob(f'save/test_{name_exp}*')[0]
#     path_csv = os.path.join(path_exp, 'roc_curve.csv')
#     df = pd.read_csv(path_csv)
#     plt.plot(df['fp_rate'], df['tp_rate'], label=name_model)
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.legend()
# plt.show()

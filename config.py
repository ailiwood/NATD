# config.py

from argparse import ArgumentParser


def get_config():
    parser = ArgumentParser(description="Training Configuration")

    parser.add_argument('--data_dir', type=str, default='Data', help='data dir')
    parser.add_argument('--data_name', type=str,
                        default='cic2018_03_02_2018', help="1.cic2018[cic2018_02_14_2018, cic2018_02_15_2018, cic2018_02_16_2018,"
                                                           "cic2018_02_20_2018, cic2018_02_21_2018, cic2018_02_22_2018, "
                                                           "cic2018_02_23_2018, cic2018_02_28_2018, cic2018_03_01_2018, "
                                                           "cic2018_03_02_2018]"
                                                           "2.unsw_nb15, "
                                                           "3.cic2019[cic2019_LDAP, cic2019_MSSQL, cic2019_NetBIOS, cic2019_Portmap,"
                                                           "cic2019_Syn,cic2019_UDP, cic2019_UDPLag")

    parser.add_argument('--log_dir', type=str, default='logs', help='the path of log_dir')
    parser.add_argument('--patience', type=int, default=2, help='the number of EarlyStopping')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size for training and evaluation'
                                                                    '1.cic2018[512], 2.unsw_nb15[512*4],3.cic2019[512*4]')
    parser.add_argument('--num_folds', type=int, default=2, help='Number of K-Fold splits')
    parser.add_argument('--num_epochs', type=int, default=60, help='Number of training epochs')
    parser.add_argument('--random_seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--input_channels', type=int, default=1, help='Number of input channels')
    parser.add_argument('--num_classes', type=int, default=None, help='(Optional) Number of classes - inferred from dataset if None')
    parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate[cic2018=5e-4, unsw_nb15=1e-4, cic2019=1e-4]')
    parser.add_argument('--optimizer', type=str, default='adam', help='Optimizer to use (adam, adamw)')
    args = parser.parse_args()
    return args

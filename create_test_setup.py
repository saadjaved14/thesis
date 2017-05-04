import helper
import glob
from os import path
from shutil import copyfile

def main():
    d_setup, __ = helper.load_setup()
    ll_test_files = glob.glob(path.join(d_setup['evaluationSetupPath'], '*test.csv'))
    v_dest = d_setup['testSetupPath']

    helper.create_directory(v_dest)

    fold_count = 0
    for fold in ll_test_files:
        v_current_destination = path.join(v_dest,'fold{0}'.format(fold_count))
        helper.create_directory(v_current_destination)
        copyfile(fold, path.join(v_current_destination, 'all_participants.csv'))
        fold_count+=1


if __name__ == '__main__':
    main()
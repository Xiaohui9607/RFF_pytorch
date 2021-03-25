from tasks.args import Args
from networks import get_model
from tasks.data import get_data

def main(opt):
    # configuration
    train_loader, test_loader, meta_data = get_data(opt)
    backbone = get_model(opt, meta_data['n_class'])




    # train & test
    pass

if __name__ == '__main__':
    opt = Args().update_args()
    main(opt)
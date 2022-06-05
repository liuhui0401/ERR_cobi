from options.errnet.train_options import TrainOptions
from engine import Engine_botnet
import torch.backends.cudnn as cudnn
import data.reflect_dataset as datasets

opt = TrainOptions().parse()

opt.isTrain = False
cudnn.benchmark = True
opt.no_log = True
opt.display_id=0
opt.verbose = False

opt.inet = 'errnet_botnet'
engine = Engine_botnet(opt)
test_ref_dataset = datasets.RealDataset('./my_test_dataset')

test_ref_dataloader = datasets.DataLoader(
                    test_ref_dataset, batch_size=1, shuffle=False,
                    num_workers=0, pin_memory=True)

res = engine.test(test_ref_dataloader, savedir="./test_tmp")
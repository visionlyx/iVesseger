import numpy
from net.HCS_Net.channel_net import *
from torch.autograd import Variable
from utils.morphology_op import *
from PyQt5.QtCore import QThread, pyqtSignal

class Refine_Seg(QThread):
    image_size = 0
    temp_label = np.zeros([image_size, image_size, image_size], dtype=np.uint8)
    fp_point_volume = list()
    fn_point_volume = list()
    seg_output = pyqtSignal(numpy.ndarray)

    def __init__(self):
        super(Refine_Seg, self).__init__()

    def run(self):
        self.refine_image_seg(self.image_size, self.temp_image, self.temp_label, self.fp_point_volume, self.fn_point_volume)

    def refine_image_seg(self, image_size, temp_image, temp_label, fp_point_volume, fn_point_volume):
        model_path = 'logs/refine_seg/HCS_Net.pth'
        os.environ["CUDA_VISIBLE_DEVICES"] = '0'
        net = HCS_Net(2, 1, image_size)

        checkpoint = torch.load(model_path)
        net.load_state_dict(checkpoint['model'])
        net = net.eval()
        net = nn.DataParallel(net)
        net = net.cuda()

        percentage1 = 0.0001
        percentage2 = 0.0003
        image = random_clip(temp_image, percentage1, percentage2)

        image_morph = morphology_op(image, fp_point_volume, fn_point_volume, image_size)

        image = np.array(image_morph, dtype=np.float32)
        thick_image = np.array(temp_label, dtype=np.float32)

        image = np.transpose((image - image.min()) / (image.max() - image.min() + 0.00001), (0, 1, 2))
        thick_image = np.transpose((thick_image - thick_image.min()) / (thick_image.max() - thick_image.min() + 0.00001), (0, 1, 2))

        image = image[np.newaxis, :]
        thick_image = thick_image[np.newaxis, :]

        test_image = Variable(torch.from_numpy(image).type(torch.FloatTensor)).cuda()
        test_thick_image = Variable(torch.from_numpy(thick_image).type(torch.FloatTensor)).cuda()

        test_image = test_image.unsqueeze(0)
        test_thick_image = test_thick_image.unsqueeze(0)

        test_image = torch.cat([test_image, test_thick_image], dim=1)
        r_image = net(test_image)
        r_image = r_image.cpu()
        r_image = r_image.squeeze(0)
        r_image = r_image.detach().numpy()
        r_image = np.where(r_image < 0.5, 0, 1)
        r_image = r_image * 255

        out_image = np.array(r_image, dtype=np.uint8)
        out_image = out_image.squeeze(0)

        image_morph_list = list()
        image_morph_list.append(image_morph)
        image_morph_list.append(out_image)
        image_morph_list = np.array(image_morph_list)

        self.seg_output.emit(image_morph_list)


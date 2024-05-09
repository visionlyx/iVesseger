import numpy
from net.U_Net.models import *
from utils.image_filter import *
from torch.autograd import Variable
from PyQt5.QtCore import QThread, pyqtSignal

class Thick_Seg(QThread):
    image_size = 0
    temp_image = np.zeros([image_size, image_size, image_size], dtype=np.uint16)
    seg_output = pyqtSignal(numpy.ndarray)

    def __init__(self):
        super(Thick_Seg, self).__init__()

    def run(self):
        self.thick_image_seg(self.temp_image)

    def thick_image_seg(self, temp_image):
        model_path = 'logs/thick_seg/U_Net.pth'
        os.environ["CUDA_VISIBLE_DEVICES"] = '0'
        net = UNet3D(1, 1, 64, layer_order='cbr')

        checkpoint = torch.load(model_path)
        net.load_state_dict(checkpoint['model'])
        net = net.eval()
        net = nn.DataParallel(net)
        net = net.cuda()

        percentage1 = 0.0001
        percentage2 = 0.0003
        image = random_clip(temp_image, percentage1, percentage2)

        image = np.array(image, dtype=np.float32)
        image = np.transpose((image - image.min()) / (image.max() - image.min()), (0, 1, 2))
        image = image[np.newaxis, :]

        test_image = Variable(torch.from_numpy(image).type(torch.FloatTensor)).cuda()
        test_image = test_image.unsqueeze(0)

        r_image = net(test_image)
        r_image = r_image.cpu()
        r_image = r_image.squeeze(0)
        r_image = r_image.detach().numpy()
        r_image = np.where(r_image < 0.5, 0, 1)
        r_image = r_image * 255

        out_image = np.array(r_image, dtype=np.uint8)
        out_image = out_image.squeeze(0)

        self.seg_output.emit(out_image)








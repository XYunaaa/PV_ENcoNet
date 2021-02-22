from .detector3d_template import Detector3DTemplate
from .PartA2_net import PartA2Net
from .point_rcnn import PointRCNN
from .pointpillar import PointPillar
from .pv_rcnn import PVRCNN
from .second_net import SECONDNet
from .randla_point_rcnn import RandLaPointRCNN
from .pointencoding_pointpillar import PointEncoding_PointPillar

__all__ = {
    'Detector3DTemplate': Detector3DTemplate,
    'SECONDNet': SECONDNet,
    'PartA2Net': PartA2Net,
    'PVRCNN': PVRCNN,
    'PointPillar': PointPillar,
    'PointRCNN': PointRCNN,
    'RandLaPointRCNN': RandLaPointRCNN,
    'PointEncoding_PointPillar': PointEncoding_PointPillar
}


def build_detector(model_cfg, num_class, dataset):

    model = __all__[model_cfg.NAME](
        model_cfg=model_cfg, num_class=num_class, dataset=dataset
    )
    return model

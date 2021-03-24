"""recognizer3d"""
from ..builder import RECOGNIZERS
from .base import BaseRecognizer


@RECOGNIZERS.register_module
class Recognizer3D(BaseRecognizer):
    """class for recognizer3d"""
    def __init__(self,
                 backbone,
                 cls_head,
                 fcn_testing=False,
                 train_cfg=None,
                 test_cfg=None):
        super(Recognizer3D, self).__init__(backbone, cls_head)
        self.fcn_testing = fcn_testing
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def forward_train(self, imgs, labels, **kwargs):
        """train"""
        #  imgs: [B clips C T H W]
        #  imgs: [Bxclips C T H W]
        imgs = imgs.reshape((-1, ) + imgs.shape[2:])

        x = self.extract_feat(imgs)
        losses = dict()
        if self.with_cls_head:
            cls_score = self.cls_head(x)
            gt_label = labels.squeeze()
            loss_cls = self.cls_head.loss(cls_score, gt_label)
            losses.update(loss_cls)

        return losses

    def forward_test(self, imgs, return_numpy, **kwargs):
        """test"""
        #  imgs: [B Clips C T H W]
        #  imgs: [BxClips C T H W]
        imgs = imgs.reshape((-1, ) + imgs.shape[2:])

        x = self.extract_feat(imgs)

        if self.with_cls_head:
            cls_score = self.cls_head(x)
            cls_score = self.average_clip(cls_score)

        if return_numpy:
            return cls_score.cpu().numpy()
        else:
            # for dp model gpu gather
            return cls_score

def build_arch(cfg):
    repo = cfg.pop('repo', None)
    assert repo is not None
    if repo == 'mmcls':
        from mmcls.models import build_classifier
        return build_classifier(cfg)
    elif repo == 'mmdet':
        from mmdet.models import build_detector
        return build_detector(cfg)
    elif repo == 'bipc':
        from bipc.models import build_classifier
        return build_classifier(cfg)
    elif repo == 'binlp':
        from binlp.models import build_classifier
        return build_classifier(cfg)
    elif repo == 'bispeech':
        from bispeech.models import build_classifier
        return build_classifier(cfg)
    else:
        raise NotImplementedError()
    
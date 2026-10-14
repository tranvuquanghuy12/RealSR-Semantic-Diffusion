import logging
logger = logging.getLogger('base')


def create_model(opt):
    model = opt['model']

    if model == 'latent_diffusion':
        from .latent_diffusion_model import LatentDiffusionModel as M
    elif model == 'ldm_v1':
        from .ldm_v1 import LDMV1 as M
    elif model == 'ldm_v2':
        from .ldm_v2 import LatentDiffusionModel as M
    elif model == 'ldm_v3':
        from .ldm_v3 import LatentDiffusionModel as M
    else:
        raise NotImplementedError('Model [{:s}] not recognized.'.format(model))
    m = M(opt)
    logger.info('Model [{:s}] is created.'.format(m.__class__.__name__))
    return m

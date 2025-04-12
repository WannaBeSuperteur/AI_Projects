# Modified Loss implementation from https://github.com/genforce/genforce/blob/master/runners/losses/logistic_gan_loss.py
# Modified Train Process implementation from https://github.com/genforce/genforce/blob/master/runners/stylegan_runner.py
# Additional Ref: https://github.com/genforce/genforce/blob/master/configs/stylegan_demo.py


import torch
import torch.nn.functional as F

ORIGINAL_HIDDEN_DIMS_Z = 512
PROPERTY_DIMS_Z = 5           # eyes, hair_color, hair_length, mouth, pose
TRAIN_BATCH_SIZE = 16


def compute_grad_penalty(images, scores):
    """Computes gradient penalty."""
    image_grad = torch.autograd.grad(
        outputs=scores.sum(),
        inputs=images,
        create_graph=True,
        retain_graph=True)[0].view(images.shape[0], -1)
    penalty = image_grad.pow(2).sum(dim=1).mean()
    return penalty


def compute_d_loss(generator, discriminator, data, gen_train_args, dis_train_args, r1_gamma, r2_gamma):
    """Computes loss for discriminator."""

    reals = data['image']
    labels = data['label']
    reals.requires_grad = True

    latents = torch.randn(reals.shape[0], ORIGINAL_HIDDEN_DIMS_Z).cuda()
    latents.requires_grad = True
    # TODO: Use random labels.
    fakes = generator(latents, label=labels, **gen_train_args)['image']
    real_scores = discriminator(reals, label=labels, **dis_train_args)
    fake_scores = discriminator(fakes, label=labels, **dis_train_args)

    d_loss = F.softplus(fake_scores).mean()
    d_loss += F.softplus(-real_scores).mean()

    real_grad_penalty = torch.zeros_like(d_loss)
    fake_grad_penalty = torch.zeros_like(d_loss)
    if r1_gamma:
        real_grad_penalty = compute_grad_penalty(reals, real_scores)
    if r2_gamma:
        fake_grad_penalty = compute_grad_penalty(fakes, fake_scores)

    return (d_loss +
            real_grad_penalty * (r1_gamma * 0.5) +
            fake_grad_penalty * (r2_gamma * 0.5))


def compute_g_loss(generator, discriminator, data, gen_train_args, dis_train_args):  # pylint: disable=no-self-use
    """Computes loss for generator."""
    # TODO: Use random labels.

    batch_size = data['image'].shape[0]
    labels = data['label']

    latents = torch.randn(batch_size, ORIGINAL_HIDDEN_DIMS_Z).cuda()
    fakes = generator(latents, label=labels, **gen_train_args)['image']
    fake_scores = discriminator(fakes, label=labels, **dis_train_args)

    g_loss = F.softplus(-fake_scores).mean()
    return g_loss


def set_model_requires_grad(model, model_name, requires_grad):
    """Sets the `requires_grad` configuration for a particular model."""
    for param in model.parameters():
        param.requires_grad = requires_grad


def moving_average_model(model, avg_model, beta=0.999):
    """Moving average model weights.

    This trick is commonly used in GAN training, where the weight of the
    generator is life-long averaged

    Args:
        model: The latest model used to update the averaged weights.
        avg_model: The averaged model weights.
        beta: Hyper-parameter used for moving average.
    """
    model_params = dict(model.named_parameters())
    avg_params = dict(avg_model.named_parameters())

    assert len(model_params) == len(avg_params)
    for param_name in avg_params:
        assert param_name in model_params
        avg_params[param_name].data = (
                avg_params[param_name].data * beta +
                model_params[param_name].data * (1 - beta))


def train_step(generator, generator_smooth, discriminator, data, gen_train_args, dis_train_args,
               r1_gamma, r2_gamma, g_smooth_img):

    # Update discriminator.
    set_model_requires_grad(discriminator, 'discriminator', True)
    set_model_requires_grad(generator, 'generator', False)

    d_loss = compute_d_loss(generator, discriminator, data, gen_train_args, dis_train_args, r1_gamma, r2_gamma)
    discriminator.optimizer.zero_grad()
    d_loss.backward()
    discriminator.optimizer.step()

    # Life-long update for generator.
    beta = 0.5 ** (TRAIN_BATCH_SIZE / g_smooth_img)
    moving_average_model(model=generator, avg_model=generator_smooth, beta=beta)

    # Update generator.
    set_model_requires_grad(discriminator, 'discriminator', False)
    set_model_requires_grad(generator, 'generator', True)

    g_loss = compute_g_loss(generator, discriminator, data, gen_train_args, dis_train_args)
    generator.optimizer.zero_grad()
    g_loss.backward()
    generator.optimizer.step()


# 모델 Fine Tuning 실시
# Create Date : 2025.04.12
# Last Update Date : -

# Arguments:
# - restructured_generator     (nn.Module)  : StyleGAN 모델의 새로운 구조의 Generator
# - restructured_discriminator (nn.Module)  : StyleGAN 모델의 새로운 구조의 Discriminator
# - stylegan_ft_loader         (DataLoader) : StyleGAN Fine-Tuning 용 데이터셋의 Data Loader

# Returns:
# - fine_tuned_generator     (nn.Module) : Fine-Tuning 된 StyleGAN 모델의 Generator
# - fine_tuned_discriminator (nn.Module) : Fine-Tuning 된 StyleGAN 모델의 Discriminator

def run_fine_tuning(restructured_generator, restructured_discriminator, stylegan_ft_loader):

    gen_train_args = dict(w_moving_decay=0.995, style_mixing_prob=0.9,
                          trunc_psi=1.0, trunc_layers=0, randomize_noise=True)
    dis_train_args = dict()

    r1_gamma = 10.0
    r2_gamma = 0.0
    g_smooth_img = 10000

    raise NotImplementedError
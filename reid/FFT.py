from math import sqrt
import torch.fft
import torch



def frft2d(x, alpha):
    frft_x = frft(x, alpha, dim=1)
    frft_y = frft(frft_x, alpha, dim=0)
    return frft_y
def frft(x, alpha, dim):
    N = x.size(dim)
    shft = torch.fft.fftshift(torch.arange(-N // 2, N // 2, dtype=torch.float32, device=x.device))
    phase = torch.exp(-1j * torch.pi * alpha / N * shft ** 2)
    if dim == 1:
        phase = phase.unsqueeze(0).unsqueeze(-1)
    if dim == 0:
        phase = phase.unsqueeze(-1).unsqueeze(-1)


    x_frft = torch.fft.fft(x * phase, dim=dim, norm='ortho')
    x_frft = torch.fft.ifft(x_frft * phase, dim=dim, norm='ortho')
    return torch.fft.fftshift(x_frft)
def ifrft2d(x, alpha):
    ifrft_x = ifrft(x, alpha, dim=1)
    ifrft_y = ifrft(ifrft_x, alpha, dim=0)
    return ifrft_y
def ifrft(x, alpha, dim):
    return frft(x, -alpha, dim)

def amplitude_spectrum_mix(img1, img2, alpha, ratio=1.0, frft_alpha=0.073):
    """Input image size: ndarray of [H, W, C]"""

    lam = torch.rand(1).item() * alpha
    assert img1.shape == img2.shape

    h, w, c = img1.shape
    h_crop = int(h * sqrt(ratio))
    w_crop = int(w * sqrt(ratio))
    h_start = h // 2 - h_crop // 2
    w_start = w // 2 - w_crop // 2

    img1_fft = frft2d(img1, alpha = frft_alpha)
    img2_fft = frft2d(img2, alpha = frft_alpha)

    img1_abs, img1_pha = torch.abs(img1_fft), torch.angle(img1_fft)
    img2_abs, img2_pha = torch.abs(img2_fft), torch.angle(img2_fft)

    img1_abs = torch.fft.fftshift(img1_abs, dim=(0, 1))
    img2_abs = torch.fft.fftshift(img2_abs, dim=(0, 1))

    img1_abs_ = torch.clone(img1_abs)
    img2_abs_ = torch.clone(img2_abs)

    img1_abs[h_start:h_start + h_crop, w_start:w_start + w_crop] = \
        lam * img2_abs_[h_start:h_start + h_crop, w_start:w_start + w_crop] + (1 - lam) * img1_abs_[
                                                                                          h_start:h_start + h_crop,
                                                                                          w_start:w_start + w_crop]

    img1_abs = torch.fft.ifftshift(img1_abs, dim=(0, 1))
    img2_abs = torch.fft.ifftshift(img2_abs, dim=(0, 1))
    img_src_random = img1_abs * (torch.exp(1j * img1_pha))

    img_src_random = ifrft2d(img_src_random, alpha = frft_alpha)
    img_src_random = torch.real(img_src_random)

    return img_src_random
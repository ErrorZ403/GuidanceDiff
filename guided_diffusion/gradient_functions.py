import torch

def take_grad(operators, x, x_hat, measurement):
    for operator in operators:
        x_hat = operator(x_hat)
    
    difference = x_hat - measurement
    loss_value = torch.linalg.norm(difference)
    gradient = torch.autograd.grad(outputs=loss_value, inputs=x)[0]

    return gradient

def take_grad_ref(operators, x, x_hat, measurement, features=None):
    for operator in operators:
        x_hat = operator(x_hat)
    
    difference_pixel = x_hat - measurement
    loss_value_pixel = torch.linalg.norm(difference_pixel)
    #x_freq, y_freq = self.get_high_pass(x_hat, measurement)

    loss_value_ref = 0.0
    if features:
        difference_ref = features['dense_features1'] - features['dense_features2']
        loss_value_ref = torch.linalg.norm(difference_ref)

    loss_value = loss_value_pixel + 0.05*loss_value_ref

    gradient = torch.autograd.grad(outputs=loss_value, inputs=x)[0]

    return gradient

def take_grad_texture(operators, x, x_hat, y):
    x_hat_deg = x_hat
    for operator in operators:
        x_hat_deg = operator(x_hat_deg)
    
    difference_pixel = x_hat_deg - y
    loss_value_pixel = torch.linalg.norm(difference_pixel)
    #x_freq, y_freq = self.get_high_pass(x_hat, measurement)

    #difference_pixel_2 = x_hat - y_hat
    #loss_value_pixel_2 = torch.linalg.norm(difference_pixel_2)

    loss_value = loss_value_pixel #+ 0.2*loss_value_pixel_2

    gradient = torch.autograd.grad(outputs=loss_value, inputs=x)[0]

    return gradient
def take_grad_feedbackSR(operators, inverse_operators, x, x_hat, y):
    for operator in operators:
        x_hat_lq = operator(x_hat)
    for operator in inverse_operators:
        y_hq = operator(y)
    
    difference_lq = x_hat_lq - y
    difference_hq = x_hat - y_hq

    loss_value_lq, loss_value_hq = torch.linalg.norm(difference_lq), torch.linalg.norm(difference_hq)

    loss_value = loss_value_lq - 0.1*loss_value_hq

    gradient = torch.autograd.grad(outputs=loss_value, inputs=x)[0]

    return gradient

def take_grad_rest(x, x_hat, y, y_rest):
    difference_lq = x_hat - y
    difference_hq = x_hat - y_rest
    loss_value_lq = torch.linalg.norm(difference_lq)
    loss_value_hq = torch.linalg.norm(difference_hq)

    x_freq, y_freq = get_high_pass(x_hat, y)
    loss_value_freq = torch.linalg.norm(x_freq - y_freq)
    loss_value = loss_value_hq - 0.25*loss_value_lq #- 0.1*loss_value_freq - 0.4*loss_value_lq

    gradient = torch.autograd.grad(outputs=loss_value, inputs=x)[0]

    return gradient

def take_grad_rest_clip(x, x_hat, y, y_rest, y_rest_clip, x_hat_clip):
    #print(y_rest_clip.shape)
    difference_lq = x_hat - y
    difference_hq = x_hat - y_rest
    difference_clip = x_hat_clip - y_rest_clip
    loss_value_lq = torch.linalg.norm(difference_lq)
    loss_value_hq = torch.linalg.norm(difference_hq)
    loss_value_clip = torch.linalg.norm(difference_clip)

    loss_value = loss_value_hq + 0.25*loss_value_clip #- 0.1*loss_value_lq

    gradient = torch.autograd.grad(outputs=loss_value, inputs=x)[0]

    return gradient

def get_high_pass(self, x, y, filter_rate=0.5):
    x_freq = torch.fft.fftshift(torch.fft.fft(x))
    y_freq = torch.fft.fftshift(torch.fft.fft(y))

    h, w = x_freq.shape[2:]
    cy, cx = int(h/2), int(w/2)
    rh, rw = int(filter_rate * cy), int(filter_rate * cx)
    x_freq[..., cy-rh:cy+rh, cx-rw:cx+rw] = 0
    y_freq[..., cy-rh:cy+rh, cx-rw:cx+rw] = 0

    x_ifft = torch.abs(torch.fft.ifft(torch.fft.ifftshift(x_freq))).clamp(-1.0, 1.0)
    y_ifft = torch.abs(torch.fft.ifft(torch.fft.ifftshift(y_freq))).clamp(-1.0, 1.0)

    return x_ifft, y_ifft
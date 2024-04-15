import sys
import torch
from tqdm import tqdm as tqdm
import numpy as np
from torch import autograd as autograd
from torch import nn as nn
from torch.nn import functional as F
from torch.autograd import Variable

class Meter(object):
    """Meters provide a way to keep track of important statistics in an online manner.
    This class is abstract, but provides a standard interface for all meters to follow.
    """

    def reset(self):
        """Reset the meter to default settings."""
        pass

    def add(self, value):
        """Log a new value to the meter
        Args:
            value: Next result to include.
        """
        pass

    def value(self):
        """Get the value of the meter in the current state."""
        pass

class AverageValueMeter(Meter):
    def __init__(self):
        super(AverageValueMeter, self).__init__()
        self.reset()
        self.val = 0

    def add(self, value, n=1):
        self.val = value
        self.sum += value
        self.var += value * value
        self.n += n

        if self.n == 0:
            self.mean, self.std = np.nan, np.nan
        elif self.n == 1:
            self.mean = 0.0 + self.sum  # This is to force a copy in torch/numpy
            self.std = np.inf
            self.mean_old = self.mean
            self.m_s = 0.0
        else:
            self.mean = self.mean_old + (value - n * self.mean_old) / float(self.n)
            self.m_s += (value - self.mean_old) * (value - self.mean)
            self.mean_old = self.mean
            self.std = np.sqrt(self.m_s / (self.n - 1.0))

    def value(self):
        return self.mean, self.std

    def reset(self):
        self.n = 0
        self.sum = 0.0
        self.var = 0.0
        self.val = 0.0
        self.mean = np.nan
        self.mean_old = 0.0
        self.m_s = 0.0
        self.std = np.nan


def compute_gradient_penalty(D, real_samples, fake_samples, device):

    # Random weight term for interpolation between real and fake samples
    alpha = torch.rand(real_samples.size(0), 1, 1, 1).to(device)
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    interpolates = interpolates.to(device)
    d_interpolates = D(interpolates)
    d_interpolates = d_interpolates.to(device)
    fake = Variable(torch.FloatTensor(real_samples.shape[0], 1, 1, 1).fill_(1.0), requires_grad=False)
    fake = fake.to(device)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


class Epoch:
    def __init__(self, model, loss, metrics, stage_name, device="cpu", verbose=True):
        self.model = model
        self.loss = loss
        self.metrics = metrics
        self.stage_name = stage_name
        self.verbose = verbose
        self.device = device

        self._to_device()

    def _to_device(self):
        self.model.to(self.device)
        self.loss.to(self.device)
        for metric in self.metrics:
            metric.to(self.device)

    def _format_logs(self, logs):
        str_logs = ["{} - {:.4}".format(k, v) for k, v in logs.items()]
        s = ", ".join(str_logs)
        return s

    def batch_update(self, x, y):
        raise NotImplementedError

    def on_epoch_start(self):
        pass

    def run(self, dataloader):
        self.on_epoch_start()

        logs = {}
        loss_meter = AverageValueMeter()
        metrics_meters = {metric.__name__: AverageValueMeter() for metric in self.metrics}

        with tqdm(dataloader, desc=self.stage_name, file=sys.stdout, disable=not (self.verbose)) as iterator:
            for x, y in iterator:
                # print(x.shape)
                x, y = x.to(self.device), y.to(self.device)
                loss, y_pred = self.batch_update(x, y)
                # loss=torch.tensor(loss)
                # update loss logs
                loss_value = loss.cpu().detach().numpy()
                loss_meter.add(loss_value)
                loss_logs = {self.loss.__class__.__name__: loss_meter.mean}
                logs.update(loss_logs)

                # update metrics logs
                for metric_fn in self.metrics:
                    metric_value = metric_fn(y_pred, y).cpu().detach().numpy()
                    metrics_meters[metric_fn.__name__].add(metric_value)
                metrics_logs = {k: v.mean for k, v in metrics_meters.items()}
                logs.update(metrics_logs)

                if self.verbose:
                    s = self._format_logs(logs)
                    iterator.set_postfix_str(s)

        return logs


class TrainEpoch(Epoch):
    def __init__(self, model, discriminator, loss, metrics, g_optimizer, d_optimizer, g_loss_fn, d_loss_fn, device="cpu", verbose=True, gp_weight=10, net_d_iters=1, net_d_init_iters=100):
        super().__init__(model, loss, metrics, "train", device, verbose)
        self.discriminator = discriminator
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer
        self.g_loss_fn = g_loss_fn
        self.d_loss_fn = d_loss_fn
        self.gp_weight = gp_weight
        self.net_d_iters = net_d_iters
        self.net_d_init_iters = net_d_init_iters
        self.current_iter = 0

    def on_epoch_start(self):
        self.model.train()
        self.discriminator.train()

    def batch_update(self, x, y):
        self.current_iter += 1
        l_g_total = 0

        # Update Generator
        for p in self.discriminator.parameters():
            p.requires_grad = False  # Freeze discriminator

        self.g_optimizer.zero_grad()

        # if (self.current_iter % self.net_d_iters == 0 and self.current_iter > self.net_d_init_iters):
        prediction_a, prediction_b, prediction_c = self.model(x)

        # pixel loss
        l_g_pix = self.loss(prediction_c, y)
        l_g_total += l_g_pix

        # gan loss
        disc_output=self.discriminator(prediction_c).squeeze(1).squeeze(1)
        #disc_output = torch.sigmoid(disc_output)  
        g_loss_fake = self.g_loss_fn(disc_output, True, is_disc=False)
        l_g_total += g_loss_fake

        l_g_total.backward()
        self.g_optimizer.step()

        # Update Discriminator
        for p in self.discriminator.parameters():
            p.requires_grad = True  # Unfreeze discriminator

        self.d_optimizer.zero_grad()
        
        disc_output=self.discriminator(y).squeeze(1).squeeze(1)
        #disc_output = torch.sigmoid(disc_output)  
        real_loss = self.d_loss_fn(disc_output, True, is_disc=True)
        # prediction_a, prediction_b, prediction_c = self.model(x) 
        prediction_a, prediction_b, prediction_c = self.model(x) 
        disc_output=self.discriminator(prediction_c).squeeze(1).squeeze(1)
        #disc_output = torch.sigmoid(disc_output)  
        fake_loss = self.d_loss_fn(disc_output,False,is_disc=True)
        gradient_penalty = compute_gradient_penalty(self.discriminator, y, prediction_c, self.device)
        d_loss = real_loss + fake_loss + self.gp_weight * gradient_penalty
        d_loss.backward()
        self.d_optimizer.step()

        return l_g_total, prediction_c


class ValidEpoch(Epoch):
    def __init__(self, model, discriminator, loss, metrics, g_loss_fn, device="cpu", verbose=True):
        super().__init__(model, loss, metrics, "valid", device, verbose)
        self.discriminator = discriminator
        self.g_loss_fn = g_loss_fn

    def on_epoch_start(self):
        self.model.eval()
        self.discriminator.eval()

    def batch_update(self, x, y):
        with torch.no_grad():
            prediction_a, prediction_b, prediction_c = self.model(x)

            # pixel loss
            l_g_pix = self.loss(prediction_c, y)

            # gan loss
            disc_output=self.discriminator(prediction_c).squeeze(1).squeeze(1)
            #disc_output = torch.sigmoid(disc_output)  
            g_loss_fake = self.g_loss_fn(disc_output, True,is_disc=True)

            total_loss = l_g_pix + g_loss_fake

            return total_loss, prediction_c




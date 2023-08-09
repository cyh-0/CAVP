import numpy as np

import torch
import torchaudio.functional
import torchaudio
# import utils.audio_utils as utils

import pdb


def stft_frame_length(pr):
    return int(pr.frame_length_ms * pr.samp_sr * 0.001)


def stft_frame_step(pr):
    return int(pr.frame_step_ms * pr.samp_sr * 0.001)


def stft_num_fft(pr):
    return int(2 ** np.ceil(np.log2(stft_frame_length(pr))))


def log10(x):
    return torch.log(x) / torch.log(torch.tensor(10.0))


def db_from_amp(x, cuda=False):
    if cuda:
        return 20.0 * log10(torch.max(torch.tensor(1e-5).to("cuda"), x.float()))
    else:
        return 20.0 * log10(torch.max(torch.tensor(1e-5), x.float()))


def amp_from_db(x):
    return torch.pow(10.0, x / 20.0)


def norm_range(x, min_val, max_val):
    return 2.0 * (x - min_val) / float(max_val - min_val) - 1.0


def unnorm_range(y, min_val, max_val):
    return 0.5 * float(max_val - min_val) * (y + 1) + min_val


def normalize_spec(spec, pr):
    return norm_range(spec, pr.spec_min, pr.spec_max)


def unnormalize_spec(spec, pr):
    return unnorm_range(spec, pr.spec_min, pr.spec_max)


def normalize_phase(phase, pr):
    return norm_range(phase, -np.pi, np.pi)


def unnormalize_phase(phase, pr):
    return unnorm_range(phase, -np.pi, np.pi)


def normalize_ims(im):
    if type(im) == type(np.array([])):
        im = im.astype("float32")
    else:
        im = im.float()
    return -1.0 + 2.0 * im


def stft(samples, pr, cuda=False):
    spec_complex = torch.stft(
        samples,
        stft_num_fft(pr),
        hop_length=stft_frame_step(pr),
        win_length=stft_frame_length(pr),
    ).transpose(1, 2)

    real = spec_complex[..., 0]
    imag = spec_complex[..., 1]
    mag = torch.sqrt((real**2) + (imag**2))
    phase = utils.angle(real, imag)
    if pr.log_spec:
        mag = db_from_amp(mag, cuda=cuda)
    return mag, phase


def make_complex(mag, phase):
    return torch.cat(
        (
            (mag * torch.cos(phase)).unsqueeze(-1),
            (mag * torch.sin(phase)).unsqueeze(-1),
        ),
        -1,
    )


def istft(mag, phase, pr):
    if pr.log_spec:
        mag = amp_from_db(mag)
    # print(make_complex(mag, phase).shape)
    samples = torchaudio.functional.istft(
        make_complex(mag, phase).transpose(1, 2),
        stft_num_fft(pr),
        hop_length=stft_frame_step(pr),
        win_length=stft_frame_length(pr),
    )
    return samples


def aud2spec(sample, pr, stereo=False, norm=False, cuda=True):
    sample = sample[:, : pr.sample_len]
    spec, phase = stft(
        sample.transpose(1, 2).reshape((sample.shape[0] * 2, -1)), pr, cuda=cuda
    )
    spec = spec.reshape(sample.shape[0], 2, pr.spec_len, -1)
    phase = phase.reshape(sample.shape[0], 2, pr.spec_len, -1)
    return spec, phase


def mix_sounds(samples0, pr, samples1=None, cuda=False, dominant=False, noise_ratio=0):
    # pdb.set_trace()
    samples0 = utils.normalize_rms(samples0, pr.input_rms)
    if samples1 is not None:
        samples1 = utils.normalize_rms(samples1, pr.input_rms)

    if dominant:
        samples0 = samples0[:, : pr.sample_len]
        samples1 = samples1[:, : pr.sample_len] * noise_ratio
    else:
        samples0 = samples0[:, : pr.sample_len]
        samples1 = samples1[:, : pr.sample_len]

    samples_mix = samples0 + samples1
    if cuda:
        samples0 = samples0.to("cuda")
        samples1 = samples1.to("cuda")
        samples_mix = samples_mix.to("cuda")

    spec_mix, phase_mix = stft(samples_mix, pr, cuda=cuda)

    spec0, phase0 = stft(samples0, pr, cuda=cuda)
    spec1, phase1 = stft(samples1, pr, cuda=cuda)

    spec_mix = spec_mix[:, : pr.spec_len]
    phase_mix = phase_mix[:, : pr.spec_len]
    spec0 = spec0[:, : pr.spec_len]
    spec1 = spec1[:, : pr.spec_len]
    phase0 = phase0[:, : pr.spec_len]
    phase1 = phase1[:, : pr.spec_len]

    return utils.Struct(
        samples=samples_mix.float(),
        phase=phase_mix.float(),
        spec=spec_mix.float(),
        sample_parts=[samples0, samples1],
        spec_parts=[spec0.float(), spec1.float()],
        phase_parts=[phase0.float(), phase1.float()],
    )


def pit_loss(pred_spec_fg, pred_spec_bg, snd, pr, cuda=True, vis=False):
    # if pr.norm_spec:
    def ns(x):
        return normalize_spec(x, pr)

    # else:
    #     def ns(x): return x
    if pr.norm:
        gts_ = [[ns(snd.spec_parts[0]), None], [ns(snd.spec_parts[1]), None]]
        preds = [[ns(pred_spec_fg), None], [ns(pred_spec_bg), None]]
    else:
        gts_ = [[snd.spec_parts[0], None], [snd.spec_parts[1], None]]
        preds = [[pred_spec_fg, None], [pred_spec_bg, None]]

    def l1(x, y):
        return torch.mean(torch.abs(x - y), (1, 2))

    losses = []
    for i in range(2):
        gt = [gts_[i % 2], gts_[(i + 1) % 2]]
        fg_spec = pr.l1_weight * l1(preds[0][0], gt[0][0])
        bg_spec = pr.l1_weight * l1(preds[1][0], gt[1][0])
        losses.append(fg_spec + bg_spec)

    losses = torch.cat([x.unsqueeze(0) for x in losses], dim=0)
    if vis:
        print(losses)
    loss_val = torch.min(losses, dim=0)
    if vis:
        print(loss_val[1])
    loss = torch.mean(loss_val[0])

    return loss


def diff_loss(spec_diff, phase_diff, snd, pr, device, norm=False, vis=False):
    def ns(x):
        return normalize_spec(x, pr)

    def np(x):
        return normalize_phase(x, pr)

    criterion = torch.nn.L1Loss()

    gt_spec_diff = snd.spec_diff
    gt_phase_diff = snd.phase_diff
    criterion = criterion.to(device)

    if norm:
        gt_spec_diff = ns(gt_spec_diff)
        gt_phase_diff = np(gt_phase_diff)
        pred_spec_diff = ns(spec_diff)
        pred_phase_diff = np(phase_diff)
    else:
        pred_spec_diff = spec_diff
        pred_phase_diff = phase_diff

    spec_loss = criterion(pred_spec_diff, gt_spec_diff)
    phase_loss = criterion(pred_phase_diff, gt_phase_diff)
    loss = pr.l1_weight * spec_loss + pr.phase_weight * phase_loss
    if vis:
        print(loss)
    return loss


def audio_stft(stft, audio, pr):
    N, C, A = audio.size()
    audio = audio.view(N * C, A)
    spec = stft(audio)
    spec = spec.transpose(-1, -2)
    spec = db_from_amp(spec, cuda=True)
    spec = normalize_spec(spec, pr)
    _, T, F = spec.size()
    spec = spec.view(N, C, T, F)
    return spec


def normalize_audio(samples, desired_rms=0.1, eps=1e-4):
    # print(np.mean(samples**2))
    rms = np.maximum(eps, np.sqrt(np.mean(samples**2)))
    samples = samples * (desired_rms / rms)
    return samples


class LoggerOutput(object):
    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            self.mkdir_if_missing(os.path.dirname(fpath))
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()

    def mkdir_if_missing(self, dir_path):
        try:
            os.makedirs(dir_path)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val*weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        val = np.asarray(val)
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        if self.val is None:
            return 0.
        else:
            return self.val.tolist()

    def average(self):
        if self.avg is None:
            return 0.
        else:
            return self.avg.tolist()


class Struct:
  def __init__(self, *dicts, **fields):
    for d in dicts:
      for k, v in d.iteritems():
        setattr(self, k, v)
    self.__dict__.update(fields)

  def to_dict(self):
    return {a: getattr(self, a) for a in self.attrs()}

  def attrs(self):
    #return sorted(set(dir(self)) - set(dir(Struct)))
    xs = set(dir(self)) - set(dir(Struct))
    xs = [x for x in xs if ((not (hasattr(self.__class__, x) and isinstance(getattr(self.__class__, x), property))) \
        and (not inspect.ismethod(getattr(self, x))))]
    return sorted(xs)

  def updated(self, other_struct_=None, **kwargs):
    s = copy.deepcopy(self)
    if other_struct_ is not None:
      s.__dict__.update(other_struct_.to_dict())
    s.__dict__.update(kwargs)
    return s

  def copy(self):
    return copy.deepcopy(self)

  def __str__(self):
    attrs = ', '.join('%s=%s' % (a, getattr(self, a)) for a in self.attrs())
    return 'Struct(%s)' % attrs


class Params(Struct):
  def __init__(self, **kwargs):
    self.__dict__.update(kwargs)


def normalize_rms(samples, desired_rms=0.1, eps=1e-4):
  rms = torch.max(torch.tensor(eps), torch.sqrt(
      torch.mean(samples**2, dim=1)).float())
  samples = samples * desired_rms / rms.unsqueeze(1)
  return samples


def normalize_rms_np(samples, desired_rms=0.1, eps=1e-4):
  rms = np.maximum(eps, np.sqrt(np.mean(samples**2, 1)))
  samples = samples * (desired_rms / rms)
  return samples


def angle(real, imag): 
  return torch.atan2(imag, real)


def atleast_2d_col(x):
  x = np.asarray(x)
  if np.ndim(x) == 0:
    return x[np.newaxis, np.newaxis]
  if np.ndim(x) == 1:
    return x[:, np.newaxis]
  else:
    return x
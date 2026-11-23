"""Fast CPU tests: model shapes, padding/inference helpers, EPE / Fl-all, flow file I/O, flow colouring."""
import cv2
import numpy as np
import pytest
import torch
from torchvision.utils import flow_to_image

import datasets
import evaluate
from models import FlowNetS, RAFTSmall, build

torch.manual_seed(0)


def test_flownets_multiscale_shapes():
    m = FlowNetS().eval()
    a, b = torch.rand(1, 3, 128, 192) * 255, torch.rand(1, 3, 128, 192) * 255
    outs = m(a, b)
    assert [tuple(o.shape) for o in outs] == [(1, 2, 128 // s, 192 // s) for s in (4, 8, 16, 32, 64)]
    assert m.predict(a, b).shape == (1, 2, 128, 192)


def test_raft_small_random_init_shapes():
    m = build("raft_small").eval()                      # weights=None: nothing is downloaded
    assert isinstance(m, RAFTSmall)
    a, b = torch.rand(1, 3, 128, 136) * 255, torch.rand(1, 3, 128, 136) * 255
    with torch.no_grad():
        preds = m(a, b, iters=3)
    assert len(preds) == 3 and preds[-1].shape == (1, 2, 128, 136)
    assert torch.isfinite(preds[-1]).all()


def test_pad_unpad_round_trip():
    x = torch.rand(1, 2, 37, 53)
    p, pad = evaluate.pad8(x)
    assert p.shape[-2] % 8 == 0 and p.shape[-1] % 8 == 0
    assert torch.equal(evaluate.unpad(p, pad), x)
    q, hw = evaluate.pad64(x)
    assert q.shape[-2:] == (64, 64) and hw == (37, 53)


class ConstFlow(torch.nn.Module):
    """Stand-in model predicting a constant flow at whatever resolution it is fed."""

    def __init__(self, u, v):
        super().__init__()
        self.uv = torch.tensor([u, v], dtype=torch.float32).view(1, 2, 1, 1)

    def predict(self, a, b, iters=None):
        return self.uv.expand(a.shape[0], 2, *a.shape[-2:]).clone()


def test_infer_rescales_flownets_flow_to_original_size():
    a = torch.zeros(1, 3, 50, 100)
    f = evaluate.infer(ConstFlow(10.0, 10.0), a, a, is_flownets=True)    # resized to 64x128 internally
    assert f.shape == (1, 2, 50, 100)
    torch.testing.assert_close(f[0, 0], torch.full((50, 100), 10.0 * 100 / 128))
    torch.testing.assert_close(f[0, 1], torch.full((50, 100), 10.0 * 50 / 64))
    g = evaluate.infer(ConstFlow(3.0, -1.0), a[..., :45, :99], a[..., :45, :99], is_flownets=False)
    assert g.shape == (1, 2, 45, 99) and torch.all(g[0, 0] == 3.0)


def test_epe_and_fl_all_metric():
    h, w = 8, 10
    gt = torch.zeros(2, h, w); gt[0] = 20.0                         # |flow| = 20 px everywhere
    valid = torch.zeros(h, w); valid[:, :5] = 1                     # only the left half is annotated
    ds = [(torch.zeros(3, h, w), torch.zeros(3, h, w), gt, valid)]

    class Pred(ConstFlow):
        def predict(self, a, b, iters=None):
            f = torch.zeros(1, 2, *a.shape[-2:]); f[:, 0] = 20.0
            f[:, 1, :4, :] = 4.0          # 4 px error (> 3 px and > 5 % of 20) on the top half
            f[:, 1, 4:, :] = 0.5          # 0.5 px error elsewhere
            return f

    r = evaluate.evaluate(Pred(0, 0), ds, "cpu", is_flownets=False, kitti=True)
    assert r["EPE"] == pytest.approx((4.0 + 0.5) / 2)
    assert r["Fl-all"] == pytest.approx(50.0)
    assert evaluate.evaluate(Pred(0, 0), ds, "cpu", False)["EPE"] == pytest.approx(2.25)


def test_flo_and_kitti_png_readers(tmp_path):
    flow = np.random.RandomState(0).randn(6, 7, 2).astype(np.float32)
    p = tmp_path / "f.flo"
    with open(p, "wb") as f:
        np.array([202021.25], np.float32).tofile(f)
        np.array([7, 6], np.int32).tofile(f)
        flow.tofile(f)
    np.testing.assert_array_equal(datasets.read_flo(str(p)), flow)

    uv = np.round(flow * 64) / 64                                  # KITTI stores flow * 64 + 2^15 in uint16
    png = np.zeros((6, 7, 3), np.uint16)
    png[..., 0], png[..., 1] = uv[..., 0] * 64 + 2 ** 15, uv[..., 1] * 64 + 2 ** 15
    png[..., 2] = 1
    cv2.imwrite(str(tmp_path / "k.png"), png[..., ::-1])            # RGB -> BGR on disk
    f2, valid = datasets.read_kitti_flow(str(tmp_path / "k.png"))
    np.testing.assert_allclose(f2, uv, atol=1e-6)
    assert np.all(valid == 1)


def test_flow_to_colour_wheel():
    flow = torch.zeros(2, 3, 3)
    flow[0, 1, 2], flow[0, 1, 0] = 5.0, -5.0                       # right vs left
    img = flow_to_image(flow)
    assert img.dtype == torch.uint8 and img.shape == (3, 3, 3)
    assert torch.all(img[:, 1, 1] == 255)                          # zero motion -> white
    assert not torch.equal(img[:, 1, 2], img[:, 1, 0])             # opposite directions -> different hues

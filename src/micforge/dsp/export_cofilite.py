"""Export a CoFi-Lite PyTorch checkpoint (.tar / .pth) to an ONNX model file."""
from __future__ import annotations

import importlib.util
import logging
from pathlib import Path
import sys

log = logging.getLogger("micforge.cofilite.export")


def _load_cofilite_class(external_dir: str | Path | None = None):
    """Dynamically load the CoFiLite model class without requiring static external imports."""
    search_dirs: list[Path] = []
    if external_dir:
        search_dirs.append(Path(external_dir).resolve())

    # Check repository root / external / CoFi-Lite
    pkg_root = Path(__file__).resolve().parents[3]
    search_dirs.append(pkg_root / "external" / "CoFi-Lite")
    search_dirs.append(pkg_root.parent / "CoFi-Lite")

    for base in search_dirs:
        candidate = base / "model" / "cofilite.py" if base.is_dir() else base
        if candidate.is_file():
            spec = importlib.util.spec_from_file_location("cofilite_dynamic_mod", str(candidate))
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                sys.modules["cofilite_dynamic_mod"] = mod
                spec.loader.exec_module(mod)
                cls = getattr(mod, "CoFiLite", None)
                if cls is not None:
                    return cls

    raise ImportError(
        "Could not locate CoFi-Lite model definition (model/cofilite.py). "
        "Please ensure the CoFi-Lite repository is cloned into external/CoFi-Lite."
    )


def export_checkpoint_to_onnx(
    checkpoint_path: str | Path,
    output_onnx_path: str | Path,
    external_dir: str | Path | None = None,
) -> Path:
    """Load a CoFi-Lite checkpoint, extract the core enhancement module, and export to ONNX."""
    import torch

    p_ckpt = Path(checkpoint_path).resolve()
    p_out = Path(output_onnx_path).resolve()
    p_out.parent.mkdir(parents=True, exist_ok=True)

    CoFiLite = _load_cofilite_class(external_dir)
    model = CoFiLite()
    ckpt = torch.load(str(p_ckpt), map_location="cpu")
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()

    class CoFiLiteCore(torch.nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base = base

        def forward(self, x):
            # x: [B, 2, T, F]
            x_ref = x.permute(0, 3, 2, 1)
            X_amp = torch.sqrt(x[:, 0].pow(2) + x[:, 1].pow(2) + 1e-12).unsqueeze(1)
            X_ric = x.div(X_amp.pow(0.7))
            X_log = torch.log10(X_amp)
            x_feat = torch.cat([X_log, X_ric], dim=1)
            x_erb = self.base.ERB.erb(X_amp)
            x_erb = torch.log10(x_erb)
            x_erb = self.base.sfe(x_erb)
            en_x1 = self.base.en_erb1(x_erb)
            en_x2 = self.base.en_erb2(en_x1)
            en_x3 = self.base.en_erb3(en_x2)
            emb_erb = self.base.inter_rnn1(en_x3)
            x_f = x_feat[..., :65]
            x_f = self.base.sfe(x_f)
            enf_x1 = self.base.en_lfb1(x_f)
            emb_lfb = self.base.inter_rnn2(enf_x1)
            emb1, emb2 = self.base.spgrnn(emb_erb, emb_lfb)
            emb_erb = self.base.inter_rnn3(emb1)
            de_x3 = self.base.de_erb3(emb_erb + en_x3)
            de_x2 = self.base.de_erb2(de_x3 + en_x2)
            de_x1 = self.base.de_erb1(de_x2 + en_x1)
            de_x1 = self.base.out_act(de_x1)
            emb_lfb = self.base.inter_rnn4(emb2)
            def_x1 = self.base.de_lfb1(emb_lfb + enf_x1)
            def_x1 = self.base.out_act(def_x1)
            est_erb = self.base.ERB.ierb(de_x1).permute(0, 3, 2, 1).contiguous()
            enh0 = x_ref * est_erb
            mask = def_x1.permute(0, 3, 2, 1)
            enh = torch.cat([enh0[:, :65] * mask, enh0[:, 65:]], dim=1)
            return enh

    core = CoFiLiteCore(model)
    dummy = torch.randn(1, 2, 16, 257)
    torch.onnx.export(
        core,
        dummy,
        str(p_out),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {2: "frames"}, "output": {2: "frames"}},
        opset_version=17,
    )
    log.info("Exported CoFi-Lite checkpoint to ONNX: %s", p_out)
    return p_out


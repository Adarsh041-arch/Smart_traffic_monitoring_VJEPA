import torch
from transformers import AutoVideoProcessor, AutoModel

class VJEPAFeatureExtractor:
    """Frozen V-JEPA 2.0 ViT-L encoder for spatiotemporal features."""

    def __init__(self, repo="facebook/vjepa2-vitl-fpc64-256",
                 device=None, use_fp16=True):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModel.from_pretrained(repo)
        self.processor = AutoVideoProcessor.from_pretrained(repo)
        self.model.eval()

        if use_fp16 and self.device == "cuda":
            self.model = self.model.half()
        self.model.to(self.device)

        for p in self.model.parameters():
            p.requires_grad = False  # frozen — feature extraction only

    @torch.no_grad()
    def extract(self, frames):
        """
        frames: list/array of T frames, shape [T, H, W, C], uint8
        returns: patch-wise feature tensor [1, num_patches, embed_dim]
        """
        inputs = self.processor(frames, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        if self.model.dtype == torch.float16:
            inputs = {k: v.half() if v.dtype == torch.float32 else v
                      for k, v in inputs.items()}

        features = self.model.get_vision_features(**inputs)
        return features.float().cpu()  # back to fp32 for downstream LSTM

    def extract_clip(self, frames, pool="mean"):
        """Returns a single pooled vector per clip — feeds LSTM directly."""
        feats = self.extract(frames)        # [1, N, D]
        if pool == "mean":
            return feats.mean(dim=1)        # [1, D]
        return feats
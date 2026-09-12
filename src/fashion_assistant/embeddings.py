"""Optional local encoders. Construction downloads weights if not cached."""
from pathlib import Path

TEXT_MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
# Immutable safetensors conversion in the original CLIP repository (refs/pr/15).
CLIP_MODEL_REVISION = "b33cedfd0df4e43b8238760678fcc89e1a0d38b3"


def checked_revision(model_id, revision, default_model, default_revision):
    if revision is not None and (not isinstance(revision, str) or not revision.strip()):
        raise ValueError("Revision must be a model commit/tag string or None, not a True/False toggle.")
    return revision or (default_revision if model_id == default_model else None)


class TextEncoder:
    def __init__(self, model_id="sentence-transformers/all-MiniLM-L6-v2", revision=None):
        from sentence_transformers import SentenceTransformer
        revision = checked_revision(model_id, revision, "sentence-transformers/all-MiniLM-L6-v2", TEXT_MODEL_REVISION)
        self.model_id, self.revision = model_id, revision
        self.model = SentenceTransformer(model_id, revision=revision, model_kwargs={"use_safetensors": True})

    def encode(self, texts):
        return self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False).tolist()


class CLIPEncoder:
    def __init__(self, model_id="openai/clip-vit-base-patch32", revision=None):
        import torch
        from transformers import CLIPModel, CLIPProcessor
        revision = checked_revision(model_id, revision, "openai/clip-vit-base-patch32", CLIP_MODEL_REVISION)
        self.model_id, self.revision = model_id, revision
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained(model_id, revision=revision, use_safetensors=True).to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_id, revision=revision)
        self.resolved_revision = getattr(self.model.config, "_commit_hash", None)

    def encode_images(self, paths, batch_size=16):
        import torch
        from PIL import Image
        vectors = []
        for start in range(0, len(paths), batch_size):
            images = []
            try:
                for path in paths[start:start+batch_size]:
                    with Image.open(Path(path)) as image:
                        images.append(image.convert("RGB"))
                inputs = self.processor(images=images, return_tensors="pt").to(self.device)
                with torch.inference_mode():
                    features = self.model.get_image_features(**inputs)
                    vectors.extend(torch.nn.functional.normalize(features, dim=-1).cpu().tolist())
            finally:
                for image in images:
                    image.close()
        return vectors

    def encode_text(self, texts):
        import torch
        inputs = self.processor(text=texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            features = self.model.get_text_features(**inputs)
        return torch.nn.functional.normalize(features, dim=-1).cpu().tolist()

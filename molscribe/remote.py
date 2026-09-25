"""Serve one MolScribe instance to other processes (RxnScribe in another container, Celery workers, ...).

Wire format: a list of base64-encoded PNG images (RGB); the answer is the list returned by
`MolScribe.predict_images`. The client side lives in rxnscribe-custom (`rxnscribe.molscribe_client`).

Nothing here is imported by `molscribe` itself, so a plain MolScribe deployment is unaffected.

FastAPI (model container)::

    from molscribe import MolScribe
    from molscribe.remote import make_router
    model = MolScribe(ckpt, device=torch.device('cuda'), precision='fp16')
    app.include_router(make_router(model))            # POST /molscribe/predict_batch

Celery (worker that owns the model)::

    from molscribe.remote import register_celery_task
    register_celery_task(celery_app, lambda: model)   # task 'molscribe.predict_batch'
"""
import base64

import cv2
import numpy as np


def encode_image(image) -> str:
    """RGB numpy array (or PIL image) -> base64 PNG (lossless)."""
    image = np.ascontiguousarray(np.asarray(image))
    if image.ndim == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode('.png', image, [cv2.IMWRITE_PNG_COMPRESSION, 1])
    if not ok:
        raise ValueError(f"cannot encode image of shape {image.shape}")
    return base64.b64encode(buf.tobytes()).decode('ascii')


def decode_image(data: str):
    """base64 PNG/JPEG -> RGB numpy array."""
    buf = np.frombuffer(base64.b64decode(data), dtype=np.uint8)
    image = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("cannot decode image")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _to_builtin(obj):
    """numpy scalars/arrays -> Python types, so the result is JSON-serialisable."""
    if isinstance(obj, dict):
        return {k: _to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_builtin(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def predict_encoded(model, encoded_images, batch_size=32, return_atoms_bonds=False, return_confidence=False):
    """Decode the images and run `model.predict_images` (thread-safe: the model serialises GPU work)."""
    images = [decode_image(data) for data in encoded_images]
    if not images:
        return []
    return _to_builtin(model.predict_images(images, batch_size=batch_size, return_atoms_bonds=return_atoms_bonds,
                                            return_confidence=return_confidence))


def make_router(model, path='/molscribe/predict_batch', max_images=4096):
    """FastAPI router: POST {"images": [b64], "batch_size": 32} -> {"predictions": [...]}.

    The handler is a plain `def`, so FastAPI runs it in its thread pool and the event loop stays free while the
    model works.
    """
    from fastapi import APIRouter, HTTPException
    from pydantic import BaseModel

    class BatchRequest(BaseModel):
        images: list
        batch_size: int = 32
        return_atoms_bonds: bool = False
        return_confidence: bool = False

    router = APIRouter()

    @router.post(path)
    def predict_batch(request: BatchRequest):
        if len(request.images) > max_images:
            raise HTTPException(status_code=413, detail=f"at most {max_images} images per request")
        try:
            predictions = predict_encoded(model, request.images, request.batch_size,
                                          request.return_atoms_bonds, request.return_confidence)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        return {'predictions': predictions}

    return router


def register_celery_task(app, get_model, name='molscribe.predict_batch', **task_options):
    """Register the task on a Celery app. `get_model()` returns the worker's MolScribe (create it lazily there)."""

    @app.task(name=name, **task_options)
    def predict_batch(encoded_images, batch_size=32, return_atoms_bonds=False, return_confidence=False):
        return predict_encoded(get_model(), encoded_images, batch_size, return_atoms_bonds, return_confidence)

    return predict_batch

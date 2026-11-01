"""Experimental decode-scoped tile batching; CUDA/TP2 not yet qualified.

Targets copied checkpoint _run_tile_tasks(self, ..., self.decode, ...).
No shared stack_tiling mutation; class installation serialized in this process.
Synchronous decode entry points and one import of this module are assumed.
"""
from contextvars import ContextVar
from functools import wraps
from threading import RLock, get_ident
import asyncio

_INSTALL_LOCK = RLock()
_ACTIVE_SCOPES = ContextVar('h3_decode_models', default=())
_MODEL_MARKER = object()
_HOST_SPEC = '_h3_context_prototype_installation'
_ENTRY_POINTS = ('decode_latent', 'decode_with_chunks')


def current_task():
    try:
        return asyncio.current_task()
    except RuntimeError:
        return None


class DecodeScope:
    __slots__ = ('model', 'thread', 'task', 'cap', 'active')

    def __init__(self, model, cap):
        self.model = model
        self.thread = get_ident()
        self.task = current_task()
        self.cap = cap
        self.active = True

    def owns(self, model):
        return (self.active and self.model is model and self.thread == get_ident()
                and self.task is current_task())


def patch_model(model):
    cls = type(model)
    with _INSTALL_LOCK:
        current = getattr(cls, "_run_tile_tasks", None)
        if current is None or getattr(current, "_context_install", None) is _MODEL_MARKER:
            return
        original = current

        @wraps(original)
        def run_tiles(self, tiles, indices, forward, stacking, cls_agg=None):
            decoder = getattr(self, "decode", None)
            is_decoder = (getattr(forward, "__self__", None) is self
                          and getattr(forward, "__func__", None) is not None
                          and getattr(forward, "__func__", None) is getattr(decoder, "__func__", None))
            allowed = not (getattr(self, "training", False)
                           and getattr(getattr(self, "decoder", None), "mask_enabled", False))
            scope = next((item for item in reversed(_ACTIVE_SCOPES.get()) if item.owns(self)), None)
            if not (scope is not None and is_decoder and allowed and cls_agg is None and indices):
                return original(self, tiles, indices, forward, stacking, cls_agg)
            output = []
            for i in range(0, len(indices), scope.cap):
                output.extend(original(self, tiles, indices[i:i + scope.cap], forward, True, None))
            return output

        run_tiles._context_install = _MODEL_MARKER
        cls._run_tile_tasks = run_tiles


def install(host_class, cap=4):
    if type(cap) is not int or not 1 <= cap <= 4:
        raise ValueError("prototype accepts explicit cap 1..4 only")

    def wrap(original):
        @wraps(original)
        def decode(self, *args, **kwargs):
            model = getattr(self, "model", None)
            if model is None or not hasattr(model, "stack_tiling"):
                return original(self, *args, **kwargs)
            patch_model(model)
            scope = DecodeScope(model, cap)
            token = _ACTIVE_SCOPES.set(_ACTIVE_SCOPES.get() + (scope,))
            try:
                return original(self, *args, **kwargs)
            finally:
                # Copies of this ContextVar may outlive the caller. Revoking
                # the shared scope makes those copies inert after return. Clear
                # strong references too: a copied context can otherwise retain
                # an entire completed asyncio task, its result, and the model.
                scope.active = False
                scope.task = None
                scope.model = None
                _ACTIVE_SCOPES.reset(token)
        return decode

    with _INSTALL_LOCK:
        installed = getattr(host_class, _HOST_SPEC, None)
        if installed is not None:
            if installed['cap'] != cap:
                raise ValueError('host class already installed with a different cap')
            if any(getattr(host_class, name, None) is not installed['entries'][name] for name in _ENTRY_POINTS):
                raise ValueError('resolved decode entry point differs from installed wrapper')
            return
        for name in _ENTRY_POINTS:
            if hasattr(host_class, name):
                setattr(host_class, name, wrap(getattr(host_class, name)))
        setattr(host_class, _HOST_SPEC,
                {'cap': cap, 'entries': {name: getattr(host_class, name, None) for name in _ENTRY_POINTS}})

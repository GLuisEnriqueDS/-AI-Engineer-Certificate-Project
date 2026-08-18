import os
from contextlib import contextmanager

from dotenv import load_dotenv

load_dotenv()

ENABLED = bool(os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY"))

if ENABLED:
    from langfuse import get_client, propagate_attributes

    _langfuse = get_client()
else:
    _langfuse = None


class _NullObservation:
    """Se entrega cuando Langfuse esta deshabilitado, para que el codigo de
    los nodos pueda llamar .update()/.score_trace() sin condicionales."""

    trace_id = None

    def update(self, *args, **kwargs):
        pass

    def score_trace(self, *args, **kwargs):
        pass


@contextmanager
def observe(as_type, name, **kwargs):
    """Span o generation de Langfuse (as_type: "span" | "generation");
    no-op si Langfuse no esta configurado."""
    if not ENABLED:
        yield _NullObservation()
        return
    with _langfuse.start_as_current_observation(as_type=as_type, name=name, **kwargs) as obs:
        yield obs


@contextmanager
def trace_session(session_id=None, user_id=None, trace_name=None):
    """Aplica session_id/user_id/trace_name a esta observacion y a todas
    las que se abran dentro (ver propagate_attributes de Langfuse)."""
    if not ENABLED:
        yield
        return
    with propagate_attributes(session_id=session_id, user_id=user_id, trace_name=trace_name):
        yield


def score(name, value, trace_id, **kwargs):
    """Score fuera de un span activo (ej. un boton de feedback que el
    usuario aprieta despues de que el turno ya termino). A diferencia de
    span.score_trace(), necesita el trace_id explicito porque no hay
    contexto de observacion abierto en ese momento."""
    if ENABLED and trace_id:
        _langfuse.create_score(name=name, value=value, trace_id=trace_id, **kwargs)


def flush():
    """Fuerza el envio del lote pendiente. Llamar al final de cada turno --
    somos procesos de corta vida (CLI/Streamlit), no un servidor de larga
    duracion donde el batching async por si solo alcanza."""
    if ENABLED:
        _langfuse.flush()

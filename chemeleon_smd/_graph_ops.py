"""Compatibility layer for the local mlx-graphs checkout used by ChemeleonSMD."""

try:
    from mlx_graphs.utils import scatter
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "ChemeleonSMD expects the local '../mlx-graphs' checkout. "
        "Install it with 'pip install -e ../mlx-graphs' before using this package."
    ) from exc


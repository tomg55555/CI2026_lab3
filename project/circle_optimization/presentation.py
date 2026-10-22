"""Readable displays for the circular-arrangement examples."""

from .problem import DistanceMatrix, validate_distances


def format_distance_matrix(
    distances: DistanceMatrix, *, max_children: int | None = None,
) -> str:
    """Label pairwise distances by child index; optionally show a square preview."""
    validate_distances(distances)
    if max_children is not None and max_children < 1:
        raise ValueError("max_children must be positive")

    n = len(distances)
    shown = n if max_children is None else min(n, max_children)
    labels = [f"Child{i}" for i in range(shown)]
    width = max(len(label) for label in labels)
    columns = labels + (["…"] if shown < n else [])
    title = f"Distance matrix ({n} × {n})"
    if shown < n:
        title += f" — first {shown} rows and columns"

    lines = [title + ":", f"{'':>{width}} | " + "  ".join(f"{label:>{width}}" for label in columns)]
    lines.append("-" * len(lines[-1]))
    for i, label in enumerate(labels):
        values = [str(distances[i][j]) for j in range(shown)]
        if shown < n:
            values.append("…")
        lines.append(f"{label:>{width}} | " + "  ".join(f"{value:>{width}}" for value in values))
    if shown < n:
        lines.append(f"{'…':>{width}} | " + "  ".join(f"{'…':>{width}}" for _ in columns))
    return "\n".join(lines)

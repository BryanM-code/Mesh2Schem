"""Read-only validation + normalisation of a dropped image.

Two kinds of image live in a build, and both come through here: the reference screenshots
(raw captures, up to three) and the AI-generated image TRELLIS actually consumes, since it
reconstructs a mesh from a 2D picture rather than from a schematic. Either way the file has
to be a real, sanely-sized picture before anything else touches it.

Same shape as `glb_validate`: hand it a path, get back (ok, size, error). Nothing here
raises, and nothing trusts the file's extension -- Pillow decides by reading the bytes.

Images are normalised to PNG on the way in, so every slot on disk holds one predictable
file (`shot-<n>.png`, `generated.png`) the same way schematics are always `original.schem`
and meshes are always `output.glb`. The user's own file is untouched, as everywhere else.
"""
from __future__ import annotations

from typing import Optional, Tuple

# Generous for a reference image, and far below Pillow's own bomb threshold. A picture
# for TRELLIS is a photo-sized thing; anything past this is a mistake or a hostile file.
MAX_PIXELS = 64_000_000          # e.g. 8000 x 8000
MIN_EDGE = 32                    # below this there is nothing for TRELLIS to reconstruct

__all__ = ["validate_image", "normalize_to_png", "MAX_PIXELS", "MIN_EDGE"]


def validate_image(path) -> Tuple[bool, Optional[Tuple[int, int]], Optional[str]]:
    """Return (ok, (width, height), error).

    ok is True only when Pillow can decode `path` as an image of a sane size. On failure
    error is a short human-readable reason and the size is None.
    """
    try:
        from PIL import Image, UnidentifiedImageError
    except ImportError:  # pragma: no cover - environment issue
        return False, None, "Pillow is not installed (pip install Pillow)."

    try:
        # verify() checks structural integrity without decoding every pixel, but it leaves
        # the file object unusable, so the size check reopens it.
        with Image.open(path) as probe:
            probe.verify()
        with Image.open(path) as img:
            width, height = img.size
            fmt = img.format or "image"
    except UnidentifiedImageError:
        return False, None, "Not an image file (or an image format we can't read)."
    except Exception as exc:
        return False, None, "Could not read this image: {}".format(exc)

    if width < MIN_EDGE or height < MIN_EDGE:
        return False, (width, height), (
            "Image is {}x{}; the minimum is {}x{}.".format(width, height, MIN_EDGE, MIN_EDGE))
    if width * height > MAX_PIXELS:
        return False, (width, height), (
            "Image is {}x{} = {:,} pixels; the limit is {:,}.".format(
                width, height, width * height, MAX_PIXELS))
    del fmt
    return True, (width, height), None


def normalize_to_png(src_path, dest_path) -> None:
    """Write `src_path` to `dest_path` as a PNG, flattening any transparency onto white.

    TRELLIS wants an opaque subject, and a stored alpha channel would only be a way for a
    thumbnail and the real input to disagree about what the picture looks like.
    """
    from PIL import Image

    with Image.open(src_path) as img:
        img = img.convert("RGBA")
        flattened = Image.new("RGB", img.size, (255, 255, 255))
        flattened.paste(img, mask=img.split()[3])
        flattened.save(dest_path, "PNG", optimize=True)

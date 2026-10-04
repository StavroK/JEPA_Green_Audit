from PIL import Image

from scripts.make_site_rgb_qc import enlarge_nearest


def test_enlarge_nearest_preserves_source_pixel_blocks():
    src = Image.new("RGB", (2, 1))
    src.putpixel((0, 0), (10, 20, 30))
    src.putpixel((1, 0), (200, 210, 220))

    out = enlarge_nearest(src, 3)

    assert out.size == (6, 3)
    assert out.getpixel((0, 0)) == (10, 20, 30)
    assert out.getpixel((2, 2)) == (10, 20, 30)
    assert out.getpixel((3, 0)) == (200, 210, 220)
    assert out.getpixel((5, 2)) == (200, 210, 220)

"""Stitch a directory of quicklook PNGs into an animated GIF."""
import argparse
import glob
import os
import sys

from PIL import Image


def collect_frames(input_dir, pattern):
    files = sorted(glob.glob(os.path.join(input_dir, pattern)))
    if not files:
        sys.exit(f"No files matching {pattern!r} in {input_dir}")
    return files


def write_gif(files, output_path, fps, loop, resize):
    duration_ms = int(round(1000.0 / fps))
    frames = []
    for f in files:
        img = Image.open(f).convert("RGBA")
        if resize is not None:
            w, h = img.size
            img = img.resize(
                (int(w * resize), int(h * resize)), Image.LANCZOS
            )
        frames.append(img.convert("P", palette=Image.ADAPTIVE))

    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=loop,
        optimize=True,
        disposal=2,
    )
    print(f"Wrote {len(frames)} frames → {output_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", help="Directory containing quicklook PNGs.")
    parser.add_argument(
        "output", help="Output .gif path."
    )
    parser.add_argument(
        "--pattern", default="*.png",
        help="Glob pattern to select frames within input_dir (default: *.png).",
    )
    parser.add_argument(
        "--fps", type=float, default=4.0,
        help="Frames per second (default: 4).",
    )
    parser.add_argument(
        "--loop", type=int, default=0,
        help="Number of loops; 0 = infinite (default: 0).",
    )
    parser.add_argument(
        "--resize", type=float, default=None,
        help="Optional scale factor to shrink frames (e.g. 0.5 for half-size).",
    )
    args = parser.parse_args()

    files = collect_frames(args.input_dir, args.pattern)
    write_gif(files, args.output, args.fps, args.loop, args.resize)


if __name__ == "__main__":
    main()

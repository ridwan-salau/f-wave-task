#!/usr/bin/env python3
"""
Clip out multiple time ranges from a video.

Usage:
  python clip_video.py input.webm output.mp4 "0:30-1:45,3:00-4:30,10:00-12:00"

Each range in the quoted string is START-END, separated by commas.
Times can be: seconds (90), MM:SS (1:30), or HH:MM:SS (0:01:30).
"""

import subprocess
import sys
import re


def parse_time(t: str) -> float:
    """Parse a time string into seconds. Supports: 90, 1:30, 0:01:30"""
    t = t.strip()
    parts = t.split(":")
    if len(parts) == 1:
        return float(parts[0])
    elif len(parts) == 2:
        return int(parts[0]) * 60 + float(parts[1])
    elif len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    else:
        raise ValueError(f"Invalid time format: {t}")


def get_video_duration(input_file: str) -> float:
    """Get video duration in seconds using ffprobe. Tries multiple methods."""
    # Method 1: container format duration
    result = subprocess.run(
        ["ffprobe", "-v", "error",
         "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", input_file],
        capture_output=True, text=True,
    )
    val = result.stdout.strip()
    if val and val != "N/A":
        return float(val)

    # Method 2: first video stream duration
    result = subprocess.run(
        ["ffprobe", "-v", "error",
         "-select_streams", "v:0",
         "-show_entries", "stream=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", input_file],
        capture_output=True, text=True,
    )
    val = result.stdout.strip()
    if val and val != "N/A":
        return float(val)

    # Method 3: decode the entire file to get duration (slow but reliable)
    result = subprocess.run(
        ["ffprobe", "-v", "error",
         "-count_frames", "-show_entries", "stream=nb_read_frames,duration",
         "-of", "csv=p=0", input_file],
        capture_output=True, text=True,
    )
    val = result.stdout.strip()
    if val and val != "N/A":
        # Use the last comma-separated value that's a number
        for part in reversed(val.replace("\n", ",").split(",")):
            part = part.strip()
            if part and part != "N/A":
                try:
                    return float(part)
                except ValueError:
                    continue

    raise RuntimeError(
        f"Could not determine duration of {input_file}. "
        "Try running: ffprobe -v error -show_entries format=duration {input_file}"
    )


def build_ffmpeg_cmd(input_file: str, output_file: str, cut_ranges: list[tuple[float, float]]):
    """Build an ffmpeg command that keeps everything EXCEPT the cut ranges."""

    # Sort cut ranges by start time
    cut_ranges = sorted(cut_ranges, key=lambda r: r[0])

    duration = get_video_duration(input_file)

    # Compute keep segments (inverse of cut ranges)
    keep_segments = []
    cursor = 0.0

    for start, end in cut_ranges:
        if start > cursor:
            keep_segments.append((cursor, start))
        cursor = max(cursor, end)

    if cursor < duration:
        keep_segments.append((cursor, duration))

    if not keep_segments:
        print("Error: nothing left to keep after cuts!")
        sys.exit(1)

    print(f"Keeping {len(keep_segments)} segment(s):")
    for i, (s, e) in enumerate(keep_segments):
        print(f"  {i+1}. {s:.1f}s – {e:.1f}s  ({e - s:.1f}s)")

    # Build filter_complex
    n = len(keep_segments)
    filter_parts = []
    concat_inputs = ""

    for i, (start, end) in enumerate(keep_segments):
        filter_parts.append(
            f"[0:v]trim=start={start}:end={end},setpts=PTS-STARTPTS[v{i}];"
        )
        filter_parts.append(
            f"[0:a]atrim=start={start}:end={end},asetpts=PTS-STARTPTS[a{i}];"
        )
        concat_inputs += f"[v{i}][a{i}]"

    filter_parts.append(f"{concat_inputs}concat=n={n}:v=1:a=1[v][a]")
    filter_complex = " ".join(filter_parts)

    cmd = [
        "ffmpeg",
        "-i", input_file,
        "-filter_complex", filter_complex,
        "-map", "[v]",
        "-map", "[a]",
        "-c:v", "libx264",
        "-c:a", "aac",
        "-y",
        output_file,
    ]

    return cmd


def main():
    if len(sys.argv) < 4:
        print(__doc__)
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    ranges_str = sys.argv[3]

    # Parse ranges like "0:30-1:45,3:00-4:30"
    cut_ranges = []
    for part in ranges_str.split(","):
        part = part.strip()
        if not part:
            continue
        match = re.match(r"(.+)-(.+)", part)
        if not match:
            print(f"Invalid range: {part}")
            sys.exit(1)
        start = parse_time(match.group(1))
        end = parse_time(match.group(2))
        cut_ranges.append((start, end))

    cmd = build_ffmpeg_cmd(input_file, output_file, cut_ranges)

    print(f"\nRunning:\n  {' '.join(cmd)}\n")
    subprocess.run(cmd)


if __name__ == "__main__":
    main()

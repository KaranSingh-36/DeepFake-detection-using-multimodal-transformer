"""Score any video file and localise the manipulated parts.

    python scripts/analyze_video.py path/to/video.mp4
    python scripts/analyze_video.py video.mp4 --checkpoint results/runs/latest/best.pt --plot timeline.png --json out.json
"""

import _bootstrap  # noqa: F401

import argparse
import json

from fusiontrace import PROJECT_ROOT
from fusiontrace.inference import VideoAnalyzer, plot_timeline


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video")
    parser.add_argument("--checkpoint", default=str(PROJECT_ROOT / "results" / "runs" / "latest" / "best.pt"))
    parser.add_argument("--stride", type=float, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--json", default=None, help="write the full result here")
    parser.add_argument("--plot", default=None, help="write a timeline PNG here")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    analyzer = VideoAnalyzer(args.checkpoint, device=args.device)
    result = analyzer.analyze(args.video, stride=args.stride, threshold=args.threshold)

    print(f"{result['label']}  score={result['video_score']:.3f}  threshold={result['threshold']:.3f}  "
          f"duration={result['duration']}s  audio={'yes' if result['has_audio'] else 'no'}  "
          f"({result['elapsed_seconds']}s on {result['device']})")
    print(f"video head max {result['max_video_head']} | audio head max {result['max_audio_head']}")
    if result["segments"]:
        print("Flagged segments:")
        for seg in result["segments"]:
            print(f"  {seg['start']:6.2f}s - {seg['end']:6.2f}s  score {seg['score']:.3f}  ({seg['modality']})")
    else:
        print("No segment above threshold.")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)
        print(f"JSON: {args.json}")
    if args.plot:
        plot_timeline(result, args.plot)
        print(f"Plot: {args.plot}")


if __name__ == "__main__":
    main()

"""Download Australian Inside Airbnb snapshots for the project."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import tempfile
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen


DATA_PAGE = "https://insideairbnb.com/get-the-data/"
DEFAULT_FILES = ("listings.csv.gz", "reviews.csv.gz", "calendar.csv.gz")
ARCHIVE_SNAPSHOTS = {
    ("sa", "barossa-valley"): (
        "2025-09-27",
        "2025-06-25",
    ),
    ("vic", "barwon-south-west-vic"): (
        "2025-09-28",
        "2025-06-25",
    ),
    ("qld", "brisbane"): (
        "2026-01-16",
        "2025-05-03",
        "2025-06-09",
        "2025-07-05",
        "2025-08-04",
        "2025-09-11",
        "2025-10-07",
        "2025-11-11",
    ),
    ("vic", "melbourne"): (
        "2025-06-10",
        "2025-09-12",
    ),
    ("nsw", "mid-north-coast"): (
        "2025-06-10",
        "2025-07-07",
        "2025-08-08",
        "2025-09-14",
        "2025-10-17",
        "2025-11-12",
    ),
    ("vic", "mornington-peninsula"): (
        "2025-06-15",
        "2025-09-18",
    ),
    ("nsw", "northern-rivers"): (
        "2025-06-17",
        "2025-09-22",
    ),
    ("qld", "sunshine-coast"): (
        "2025-04-30",
        "2025-05-31",
        "2025-06-28",
        "2025-07-30",
        "2025-08-31",
        "2025-09-30",
    ),
    ("nsw", "sydney"): (
        "2025-06-10",
        "2025-09-12",
    ),
    ("tas", "tasmania"): (
        "2025-06-09",
        "2025-09-11",
    ),
    ("wa", "western-australia"): ("2025-06-23",),
}
URL_RE = re.compile(
    r"^/australia/(?P<state>[^/]+)/(?P<city>[^/]+)/(?P<snapshot>\d{4}-\d{2}-\d{2})/data/(?P<file>[^/]+)$"
)


@dataclass(frozen=True)
class DatasetFile:
    city: str
    state: str
    snapshot: str
    file_name: str
    url: str


class LinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "a":
            return
        for key, value in attrs:
            if key == "href" and value:
                self.links.append(value)


def fetch_text(url: str) -> str:
    request = Request(url, headers={"User-Agent": "comp90051-project/0.1"})
    with urlopen(request, timeout=60) as response:
        return response.read().decode("utf-8")


def discover_files(files: set[str], since: date | None) -> list[DatasetFile]:
    parser = LinkParser()
    parser.feed(fetch_text(DATA_PAGE))

    discovered: list[DatasetFile] = []
    for link in parser.links:
        parsed = urlparse(link)
        if parsed.netloc != "data.insideairbnb.com":
            continue
        match = URL_RE.match(parsed.path)
        if not match:
            continue
        if match.group("file") not in files:
            continue

        snapshot = date.fromisoformat(match.group("snapshot"))
        if since is not None and snapshot < since:
            continue

        discovered.append(
            DatasetFile(
                city=match.group("city"),
                state=match.group("state"),
                snapshot=match.group("snapshot"),
                file_name=match.group("file"),
                url=link,
            )
        )

    return sorted(discovered, key=lambda item: (item.city, item.snapshot, item.file_name))


def archived_files(files: set[str], since: date | None) -> list[DatasetFile]:
    items: list[DatasetFile] = []

    for (state, city), snapshots in ARCHIVE_SNAPSHOTS.items():
        for snapshot in snapshots:
            if since is not None and date.fromisoformat(snapshot) < since:
                continue
            for file_name in sorted(files):
                items.append(
                    DatasetFile(
                        city=city,
                        state=state,
                        snapshot=snapshot,
                        file_name=file_name,
                        url=(
                            f"https://data.insideairbnb.com/australia/{state}/{city}/"
                            f"{snapshot}/data/{file_name}"
                        ),
                    )
                )

    return items


def unique_files(items: list[DatasetFile]) -> list[DatasetFile]:
    return sorted(
        {(item.city, item.snapshot, item.file_name): item for item in items}.values(),
        key=lambda item: (item.city, item.snapshot, item.file_name),
    )


def destination(root: Path, item: DatasetFile) -> Path:
    return root / item.city / item.snapshot / item.file_name


def download(item: DatasetFile, output_root: Path, overwrite: bool) -> Path:
    dest = destination(output_root, item)
    if dest.exists() and not overwrite:
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    request = Request(item.url, headers={"User-Agent": "comp90051-project/0.1"})
    with urlopen(request, timeout=300) as response:
        with tempfile.NamedTemporaryFile(dir=dest.parent, delete=False) as tmp:
            tmp.write(response.read())
            tmp_path = Path(tmp.name)
    tmp_path.replace(dest)
    return dest


def write_manifest(output_root: Path, items: list[DatasetFile]) -> None:
    manifest_path = output_root / "manifest.json"
    csv_path = output_root / "manifest.csv"
    manifest_path.write_text(json.dumps([asdict(item) for item in items], indent=2) + "\n")

    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(items[0]).keys()))
        writer.writeheader()
        for item in items:
            writer.writerow(asdict(item))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/raw/inside_airbnb_australia"))
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--files", nargs="+", default=list(DEFAULT_FILES))
    parser.add_argument("--current-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    since = date.today() - timedelta(days=args.days)
    files = set(args.files)
    items = discover_files(files, since)
    if not items:
        print("No matching Australian Inside Airbnb files found.", file=sys.stderr)
        return 1
    if not args.current_only:
        items = unique_files([*items, *archived_files(files, since)])

    cities = sorted({item.city for item in items})
    snapshots = sorted({item.snapshot for item in items})
    print(f"Found {len(items)} files across {len(cities)} cities and {len(snapshots)} snapshots.")
    print("Cities:", ", ".join(cities))

    if args.dry_run:
        for item in items:
            print(item.url)
        return 0

    for index, item in enumerate(items, start=1):
        path = download(item, args.output, args.overwrite)
        print(f"[{index}/{len(items)}] {path}")

    write_manifest(args.output, items)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

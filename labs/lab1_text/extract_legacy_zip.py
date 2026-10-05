"""Extract a large legacy ZIP whose 32-bit offsets wrap at four GiB.

Some Google Drive archives contain a valid central directory but omit ZIP64
metadata. Windows Explorer then reports an invalid format. This helper validates
every local header and corrects only offsets that are exactly four GiB too large.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import zipfile

LOCAL_HEADER = b"PK\x03\x04"
FOUR_GIB = 1 << 32


def extract_ruslan(archive: Path, destination: Path) -> int:
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as source:
        all_members = source.infolist()
        members = [
            member
            for member in all_members
            if member.filename.startswith("RUSLAN/") and not member.is_dir()
        ]
        with archive.open("rb") as raw:
            for member in all_members:
                raw.seek(member.header_offset)
                if raw.read(4) == LOCAL_HEADER:
                    continue
                corrected = member.header_offset - FOUR_GIB
                if corrected < 0:
                    raise zipfile.BadZipFile(f"Invalid offset for {member.filename}")
                raw.seek(corrected)
                if raw.read(4) != LOCAL_HEADER:
                    raise zipfile.BadZipFile(f"Invalid header for {member.filename}")
                member.header_offset = corrected

        # ZipFile calculated overlap-protection boundaries before offsets were
        # corrected. Rebuild them in physical archive order.
        end_offset = source.start_dir
        for member in sorted(all_members, key=lambda item: item.header_offset, reverse=True):
            member._end_offset = end_offset
            end_offset = member.header_offset

        print(f"Validated {len(all_members)} entry offsets", flush=True)
        skipped = 0
        for number, member in enumerate(members, start=1):
            output_path = destination / Path(member.filename)
            if output_path.is_file() and output_path.stat().st_size == member.file_size:
                skipped += 1
                continue
            source.extract(member, destination)
            if number % 1000 == 0:
                print(f"Extracted {number}/{len(members)}", flush=True)
        if skipped:
            print(f"Reused {skipped} already extracted files", flush=True)
    return len(members)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("destination", type=Path)
    arguments = parser.parse_args()
    count = extract_ruslan(arguments.archive, arguments.destination)
    print(f"Extraction complete: {count} files", flush=True)

#!/usr/bin/env python3
"""CLI safety net: reconcile the DB with what's on disk under data/builds/.

Usage:  python rebuild_from_disk.py

Use this when the DB and folders disagree -- e.g. collector.db was deleted, or build
folders were copied in by hand. It re-derives a row for each data/builds/<id>__<slug>/
folder (dims + hash from the schem, status from whether an output.glb exists) and never
deletes any row or file. The same logic is available in the app via the "Rebuild from
disk" admin button.
"""
import db


def main() -> None:
    db.init_db()
    summary = db.rebuild_from_disk()
    print("Rebuild complete.")
    print("  scanned folders : {}".format(summary["scanned"]))
    print("  rows added      : {}".format(summary["added"]))
    print("  rows updated    : {}".format(summary["updated"]))
    skipped = summary["skipped"]
    if skipped:
        print("  skipped ({}):".format(len(skipped)))
        for s in skipped:
            print("    - {}  ({})".format(s["folder"], s["reason"]))
    else:
        print("  skipped         : 0")


if __name__ == "__main__":
    main()

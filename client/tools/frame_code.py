#!/usr/bin/env python3
"""Decode the debug frame stamp from a screenshot.

usage: python3 tools/frame_code.py decode <png>

Finds the 2-row code strip by its red finder blocks (3 cells at each end of each row), samples the
56 x 2 data cells, rebuilds the 14-byte code and checks CRC16-CCITT.
Prints "frame=<n> hash=<hex> crc=ok|bad", then "sidecar=<json>" if <png stem>.json exists.
Exit 1 if the strip is not found or the CRC is bad. Standard library only (zlib + struct PNG reader).
Reads 8-bit RGB or RGBA, non-interlaced PNGs.
"""

import json
import os
import struct
import sys
import zlib

COLS = 56
ROWS = 2
FINDER = 3
TOTAL_COLS = COLS + 2 * FINDER
PAYLOAD = 12
CODE_BYTES = 14


def paeth(a: int, b: int, c: int) -> int:
    p = a + b - c
    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    if pb <= pc:
        return b
    return c


def read_png(path: str):
    """Return (width, height, channels, rows) with rows as bytearrays of unfiltered samples."""
    with open(path, "rb") as fh:
        data = fh.read()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("not a PNG file")
    pos = 8
    width = height = depth = ctype = interlace = None
    idat = []
    while pos + 8 <= len(data):
        length, = struct.unpack(">I", data[pos:pos + 4])
        kind = data[pos + 4:pos + 8]
        body = data[pos + 8:pos + 8 + length]
        pos += 12 + length
        if kind == b"IHDR":
            width, height, depth, ctype, _comp, _filt, interlace = struct.unpack(">IIBBBBB", body)
        elif kind == b"IDAT":
            idat.append(body)
        elif kind == b"IEND":
            break
    if width is None:
        raise ValueError("PNG has no IHDR")
    if depth != 8 or ctype not in (2, 6) or interlace != 0:
        raise ValueError("unsupported PNG (need 8-bit RGB/RGBA, non-interlaced): depth=%s color=%s interlace=%s"
                         % (depth, ctype, interlace))
    channels = 3 if ctype == 2 else 4
    raw = zlib.decompress(b"".join(idat))
    stride = width * channels
    rows = []
    prev = bytearray(stride)
    p = 0
    for _ in range(height):
        ft = raw[p]
        line = bytearray(raw[p + 1:p + 1 + stride])
        p += 1 + stride
        if ft == 0:
            pass
        elif ft == 1:
            for i in range(channels, stride):
                line[i] = (line[i] + line[i - channels]) & 0xFF
        elif ft == 2:
            for i in range(stride):
                line[i] = (line[i] + prev[i]) & 0xFF
        elif ft == 3:
            for i in range(stride):
                a = line[i - channels] if i >= channels else 0
                line[i] = (line[i] + ((a + prev[i]) >> 1)) & 0xFF
        elif ft == 4:
            for i in range(stride):
                a = line[i - channels] if i >= channels else 0
                c = prev[i - channels] if i >= channels else 0
                line[i] = (line[i] + paeth(a, prev[i], c)) & 0xFF
        else:
            raise ValueError("bad PNG filter type %d" % ft)
        rows.append(line)
        prev = line
    return width, height, channels, rows


def find_strip(width: int, height: int, channels: int, rows):
    """Bounding box of the red finder blocks -> (x0, y0, cell_w, cell_h), or None."""
    min_x = min_y = None
    max_x = max_y = -1
    for y, line in enumerate(rows):
        for x in range(width):
            i = x * channels
            if line[i] >= 200 and line[i + 1] <= 60 and line[i + 2] <= 60:
                if min_x is None or x < min_x:
                    min_x = x
                if min_x is None or x > max_x:
                    max_x = x
                if min_y is None or y < min_y:
                    min_y = y
                max_y = y
    if min_x is None or max_x < min_x:
        return None
    w = max_x - min_x + 1
    h = max_y - min_y + 1
    cell_w = w / float(TOTAL_COLS)
    cell_h = h / float(ROWS)
    if cell_w < 1.0 or abs(cell_w - cell_h) > 0.25 * cell_w:
        return None
    return min_x, min_y, cell_w, cell_h


def sample_code(width: int, height: int, channels: int, rows, strip) -> bytes:
    x0, y0, cell_w, cell_h = strip
    bits = []
    for r in range(ROWS):
        y = min(height - 1, int(y0 + (r + 0.5) * cell_h))
        line = rows[y]
        for c in range(COLS):
            x = min(width - 1, int(x0 + (FINDER + c + 0.5) * cell_w))
            i = x * channels
            lum = (line[i] + line[i + 1] + line[i + 2]) / 3.0
            bits.append(1 if lum > 127 else 0)
    out = bytearray()
    for k in range(0, len(bits), 8):
        value = 0
        for bit in bits[k:k + 8]:
            value = (value << 1) | bit
        out.append(value)
    return bytes(out)


def crc16(data: bytes) -> int:
    crc = 0xFFFF
    for b in data:
        crc ^= b << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ 0x1021) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
    return crc


def decode(path: str) -> int:
    try:
        width, height, channels, rows = read_png(path)
    except (OSError, ValueError, zlib.error, struct.error) as err:
        print("frame_code: %s: %s" % (path, err), file=sys.stderr)
        return 1
    strip = find_strip(width, height, channels, rows)
    if strip is None:
        print("frame_code: no frame stamp found in %s" % path, file=sys.stderr)
        return 1
    code = sample_code(width, height, channels, rows, strip)
    payload, stored = code[:PAYLOAD], (code[12] << 8) | code[13]
    frame = struct.unpack(">I", payload[0:4])[0]
    state_hash = struct.unpack(">Q", payload[4:12])[0]
    ok = crc16(payload) == stored
    print("frame=%d hash=%016x crc=%s" % (frame, state_hash, "ok" if ok else "bad"))
    sidecar = os.path.splitext(path)[0] + ".json"
    if os.path.exists(sidecar):
        try:
            with open(sidecar, "r", encoding="utf-8") as fh:
                print("sidecar=" + json.dumps(json.load(fh), separators=(",", ":")))
        except (OSError, ValueError) as err:
            print("frame_code: sidecar unreadable: %s" % err, file=sys.stderr)
    return 0 if ok else 1


def main(argv) -> int:
    if len(argv) == 3 and argv[1] == "decode":
        return decode(argv[2])
    print("usage: frame_code.py decode <png>", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))

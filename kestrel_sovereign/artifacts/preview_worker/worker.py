"""Container PID 1 supervises an untrusted decoder; no Core imports or secrets.

The host enforces the aggregate cgroup and wall deadline. This independent
PID-1 deadline also ends the entire container if the host connection vanishes.
Only bounded raw RGB leaves the sandbox. PNG encoding belongs to the host.
"""

import io
import json
import os
from pathlib import Path
import resource
import signal
import struct
import subprocess
import sys
import tempfile
import warnings

MAX_SOURCE = 16 * 1024 * 1024
MAX_EDGE = 1024
MAX_PIXELS = MAX_EDGE * MAX_EDGE
MAX_RAW = MAX_PIXELS * 3 + 12
MAX_SOURCE_PIXELS = 16 * 1024 * 1024


def decode():
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = MAX_SOURCE_PIXELS
    warnings.simplefilter('error', Image.DecompressionBombWarning)
    header = sys.stdin.buffer.readline(1025)
    if len(header) > 1024:
        raise ValueError
    request = json.loads(header)
    if set(request) != {'media_type', 'size'} or type(request['size']) is not int or not 0 < request['size'] <= MAX_SOURCE:
        raise ValueError
    source = sys.stdin.buffer.read(request['size'] + 1)
    if len(source) != request['size']:
        raise ValueError
    if request['media_type'] == 'application/pdf':
        with tempfile.TemporaryDirectory(dir='/tmp') as directory:
            path = Path(directory)
            (path / 'source.pdf').write_bytes(source)
            subprocess.run(['/usr/bin/pdftoppm', '-f', '1', '-singlefile', '-scale-to', str(MAX_EDGE),
                            '-r', '96', '-png', str(path / 'source.pdf'), str(path / 'page')],
                           check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5)
            target = path / 'page.png'
            if target.stat().st_size > MAX_RAW:
                raise ValueError
            with Image.open(target) as image:
                return pixels(image)
    if request['media_type'] not in {'image/png', 'image/jpeg', 'image/webp', 'image/gif'}:
        raise ValueError
    with Image.open(io.BytesIO(source)) as image:
        return pixels(image)


def pixels(image):
    if image.width <= 0 or image.height <= 0 or image.width * image.height > MAX_SOURCE_PIXELS:
        raise ValueError
    image.seek(0)
    image.thumbnail((MAX_EDGE, MAX_EDGE))
    rgb = image.convert('RGB')
    if rgb.width * rgb.height > MAX_PIXELS:
        raise ValueError
    return b'KAP1' + struct.pack('!II', rgb.width, rgb.height) + rgb.tobytes()


def main():
    # All descendants inherit these kernel limits. The cgroup additionally
    # bounds their combined memory/CPU/process tree, including Poppler.
    resource.setrlimit(resource.RLIMIT_FSIZE, (32 * 1024 * 1024,) * 2)
    resource.setrlimit(resource.RLIMIT_CPU, (4, 4))
    if sys.argv[1:] == ['--decode']:
        sys.stdout.buffer.write(decode())
        return
    signal.signal(signal.SIGALRM, lambda *_: os._exit(124))
    signal.alarm(8)
    source = sys.stdin.buffer.read(MAX_SOURCE + 1025)
    if len(source) > MAX_SOURCE + 1024:
        raise ValueError
    result = subprocess.run([sys.executable, '-I', __file__, '--decode'], input=source,
                            capture_output=True, timeout=6)
    if result.returncode != 0 or len(result.stdout) > MAX_RAW:
        raise ValueError
    sys.stdout.buffer.write(result.stdout)


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, subprocess.SubprocessError):
        # No decoder diagnostics, filenames, source or stack traces leave PID1.
        sys.stderr.write('preview refused\n')
        sys.exit(2)

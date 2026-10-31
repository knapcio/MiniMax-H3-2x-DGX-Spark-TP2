#!/usr/bin/env python3
"""Make a separate timestamped contact sheet from a verified 24 fps H3 clip."""
import argparse
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from PIL import Image, ImageDraw, ImageFont


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def make(source, destination, interval):
    source = source.resolve()
    destination = Path(os.path.abspath(destination))
    if source == destination or os.path.lexists(destination) or not isinstance(interval,int) or isinstance(interval,bool) or interval < 1:
        raise ValueError('output must be new and interval must be a positive number of seconds')
    before = digest(source)
    probe = subprocess.run(['ffprobe','-v','error','-select_streams','v:0','-show_entries',
                            'stream=width,height,avg_frame_rate,nb_frames','-of','json',str(source)],
                           check=True,capture_output=True,text=True,timeout=60)
    streams = json.loads(probe.stdout)['streams']
    if len(streams) != 1 or Fraction(streams[0]['avg_frame_rate']) != 24:
        raise ValueError('expected one constant 24 fps video stream')
    count = int(streams[0]['nb_frames'])
    if count < 1:
        raise ValueError('video has no reliable frame count')
    frame_probe = subprocess.run(['ffprobe','-v','error','-select_streams','v:0','-show_frames',
                                  '-show_entries','frame=best_effort_timestamp_time','-of','json',str(source)],
                                 check=True,capture_output=True,text=True,timeout=600)
    frame_times = [Fraction(frame['best_effort_timestamp_time']) for frame in json.loads(frame_probe.stdout)['frames']]
    if len(frame_times)!=count or any(abs(value-Fraction(i,24))>Fraction(1,1000000) for i,value in enumerate(frame_times)):
        raise ValueError('video frame timestamps are not a zero-start 24 fps sequence')
    stride = 24 * interval
    expected = (count - 1) // stride + 1
    if expected > 100:
        raise ValueError('interval would create more than 100 thumbnails')
    destination.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.h3-sheet-',dir=destination.parent) as temp:
        root=Path(temp)
        subprocess.run(['ffmpeg','-v','error','-xerror','-i',str(source),'-map','0:v:0','-an','-vf',
                        f'select=not(mod(n\\,{stride})),scale=480:-2','-fps_mode','passthrough',
                        str(root/'frame-%03d.png')],check=True,timeout=7200)
        frames=sorted(root.glob('frame-*.png'))
        if len(frames)!=expected or digest(source)!=before:
            raise ValueError('thumbnail count differs or source changed during decoding')
        with Image.open(frames[0]) as frame:
            width,height=frame.size
        columns=3 if len(frames)<=6 else 4
        rows=math.ceil(len(frames)/columns)
        canvas=Image.new('RGB',(columns*width,56+rows*(height+30)), '#111827')
        draw=ImageDraw.Draw(canvas)
        try:
            font=ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc',19)
            small=ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc',16)
        except OSError:
            font=ImageFont.load_default(); small=font
        draw.text((12,6),'Klatki w czasie — podgląd kształtu chmur i światła',font=font,fill='white')
        draw.text((12,30),'Pojedyncze klatki nie oceniają płynności ani zgodności ruchu cieni.',font=small,fill='#cbd5e1')
        for i,path in enumerate(frames):
            x=(i%columns)*width; y=56+(i//columns)*(height+30)
            with Image.open(path) as frame:
                if frame.size!=(width,height): raise ValueError('thumbnail sizes differ')
                canvas.paste(frame.convert('RGB'),(x,y))
            draw.text((x+8,y+height+5),f'{i*interval} s',font=small,fill='white')
        png=root/'sheet.png'; canvas.save(png)
        with Image.open(png) as check: check.verify()
        if digest(source)!=before: raise ValueError('source changed while producing contact sheet')
        checksum=digest(png)
        os.link(png,destination)
    return {'source':str(source),'source_sha256':before,'output':str(destination),'output_sha256':checksum,
            'sample_seconds':[i*interval for i in range(expected)],'source_frames':count,
            'caveat':'Still frames do not establish motion smoothness or cloud-shadow alignment.'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source',type=Path)
    parser.add_argument('output',type=Path)
    parser.add_argument('--interval',type=int,default=10)
    args=parser.parse_args()
    try:
        print(json.dumps(make(args.source,args.output,args.interval),indent=2))
        return 0
    except (OSError,ValueError,KeyError,TypeError,ZeroDivisionError,subprocess.SubprocessError) as exc:
        print('h3-contact-sheet:',exc,file=sys.stderr)
        return 1


if __name__=='__main__': raise SystemExit(main())

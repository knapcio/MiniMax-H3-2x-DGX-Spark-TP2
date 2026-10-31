#!/usr/bin/env python3
"""Compare matched 30 s locked-camera renders at window sizes 277 and 175."""
import argparse
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys


HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('compare_speed',HERE/'compare_speed.py')
speed=importlib.util.module_from_spec(spec); spec.loader.exec_module(speed)
BASE='quality30-camera-locked-960x544-20261031'
SMALL='quality30-camera-locked-win175-960x544-20261031'
TAG='d30_960x544_s20_seed1101_fl2va'


def compare(root):
    base=speed.read_run(root,BASE,TAG)
    small=speed.read_run(root,SMALL,TAG)
    a,b=base['timing'],small['timing']
    for key in ('duration','effective_duration','size','steps','seed','task','image','frame_indices','prompt','audio_reference'):
        if a.get(key)!=b.get(key): raise ValueError('requests differ in '+key)
    if (a.get('duration'),a.get('size'),a.get('steps'),a.get('seed'),a.get('task'))!=(30,'960x544',20,1101,'fl2va'):
        raise ValueError('unexpected experiment shape or task')
    x,y=dict(a['extra_params']),dict(b['extra_params'])
    if x.get('long_video') is not True or x.get('long_video_mode')!='continuation' or x.get('preencode_mp4') is not True:
        raise ValueError('unexpected streaming mode')
    px,py=x.pop('continuation_prompts',None),y.pop('continuation_prompts',None)
    if x.pop('continuation_window_frames',277)!=277 or y.pop('continuation_window_frames',None)!=175:
        raise ValueError('unexpected continuation window sizes')
    prompt=a['prompt'].strip()
    first=prompt.rstrip('. ')+'. One single continuous take from a steady camera, no cuts, no transitions.'
    rest='The same continuous shot goes on without any cut or transition, the camera remains locked on a tripod: '+prompt
    if px!=[first,rest,rest] or py!=[first]+[rest]*4 or x!=y:
        raise ValueError('controls differ beyond window size and repeated continuation prompts')
    hashes=[(root/name/'source.sha256').read_text().strip() for name in (BASE,SMALL)]
    pids=[(root/name/'service-pids.txt').read_text().strip() for name in (BASE,SMALL)]
    if len(set(hashes))!=1 or not re.fullmatch('[0-9a-f]{64}',hashes[0]): raise ValueError('source-image checksums differ')
    if len(set(pids))!=1 or not re.fullmatch(r'[1-9][0-9]* [1-9][0-9]*',pids[0]): raise ValueError('service PIDs differ')
    if len(base['diffuse_seconds'])<3 or len(small['diffuse_seconds'])<5: raise ValueError('missing per-window timings')
    aw,bw=float(a['elapsed_s']),float(b['elapsed_s'])
    return {'source_image_sha256':hashes[0],'head_agent_pids':pids[0],
            'wall_seconds':{'277':aw,'175':bw},'wall_change_percent':round((bw/aw-1)*100,2),
            'diffuse_seconds_per_window':{'277':base['diffuse_seconds'],'175':small['diffuse_seconds']},
            'memory':{'277':base['memory'],'175':small['memory']},
            'quality_signals':{'277':base['quality_signals'],'175':small['quality_signals']},
            'caveat':'One pair with the same service PIDs; child-worker continuity and identical runtime state are not established. '
                     'The 10-step request runs between these two controls. '
                     'Smaller windows change temporal context and increase the number of seams; playback quality requires review.'}


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('root',type=Path); args=parser.parse_args()
    try: print(json.dumps(compare(args.root),indent=2)); return 0
    except (OSError,ValueError,KeyError,TypeError,ZeroDivisionError,subprocess.SubprocessError) as exc:
        print('compare-window175:',exc,file=sys.stderr); return 1


if __name__=='__main__': raise SystemExit(main())

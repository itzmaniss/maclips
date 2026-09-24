"""Static per-shot face crops and splits; no speaker attribution or mouth signal."""


def plan_layouts(analysis):
    plans={}
    for cid,result in analysis.items():
        segments=[];has_split=False;has_face=False
        for shot in result['shots']:
            tracks=[t for t in result['tracks'] if t['shot_index']==shot['shot_index']]
            tracks.sort(key=lambda t:-(t['last_s']-t['first_s']))
            segment={'start':shot['start_s'],'end':shot['end_s'],'kind':'letterbox'}
            if len(tracks)>=2:
                # Explicit attribution-free choice: duration, never speaking time.
                chosen=tracks[:2]
                # Two fragments that never coexisted do not establish a two-face shot.
                coexist=min(t['last_s'] for t in chosen)-max(t['first_s'] for t in chosen)
                if coexist>=1:
                    segment.update(kind='split',boxes=[t['median_box'] for t in chosen])
                    has_split=True;has_face=True
                else:
                    segment.update(kind='face-centred',x=tracks[0]['median_box'][0]+tracks[0]['median_box'][2]/2)
                    has_face=True
            elif tracks:
                box=tracks[0]['median_box'];segment.update(kind='face-centred',x=box[0]+box[2]/2);has_face=True
            segments.append(segment)
        options={}
        if has_face:
            name='split' if has_split else 'face-centred'
            options[name]={'kind':name,'segments':segments}
        options['letterbox']={'kind':'letterbox'}
        plans[cid]=options
    return plans

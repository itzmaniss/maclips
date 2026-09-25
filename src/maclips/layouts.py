"""Static per-shot face crops and splits; no speaker attribution or mouth signal."""

SPLIT_FACE_MARGIN=.25  # face widths kept clear of the divider between panels [I]


def split_divider(boxes):
    """Normalised x where the two panel crops meet, or None when faces are too close.

    Crops may not overlap horizontally, so each face (plus a quarter face width
    of margin) must lie on its own side of the midpoint between face centres.
    Faces stacked in one column never satisfy this and are not split.
    """
    left,right=sorted(boxes,key=lambda b:b[0]+b[2]/2)
    divider=(left[0]+left[2]/2+right[0]+right[2]/2)/2
    if left[0]+left[2]*(1+SPLIT_FACE_MARGIN)>divider or right[0]-right[2]*SPLIT_FACE_MARGIN<divider:
        return None
    return divider


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
                boxes=[t['median_box'] for t in chosen]
                if coexist>=1 and split_divider(boxes) is not None:
                    segment.update(kind='split',boxes=boxes)
                    has_split=True;has_face=True
                else:
                    # Fragments, or faces too close for separate panels: longest track.
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

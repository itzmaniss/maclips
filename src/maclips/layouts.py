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


def _face(box):
    # Centre for the crop, plus the box so the punch-in zoom can keep the face in frame.
    return {'kind':'face-centred','x':box[0]+box[2]/2,'y':box[1]+box[3]/2,'box':list(box)}


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
                    segment.update(_face(tracks[0]['median_box']))
                    has_face=True
            elif tracks:
                segment.update(_face(tracks[0]['median_box']));has_face=True
            segments.append(segment)
        options={}
        if has_face:
            name='split' if has_split else 'face-centred'
            options[name]={'kind':name,'segments':segments}
        options['letterbox']={'kind':'letterbox'}
        plans[cid]=options
    return plans


def speaker_changes(words,turns):
    """Source times where the S4 speaker label changes between consecutive words.

    Words are labelled from the window's diarized turns with `assign_speakers`,
    the overlap rule S4 itself uses. Words inside no turn stay unlabelled and
    never trigger a change. Labels are window-local (§2.2 item 1).
    """
    if not turns:
        return []
    from .diarize import DiarizationResult,Turn,assign_speakers
    from .transcribe import Word
    low=min(t['start'] for t in turns);high=max(t['end'] for t in turns)
    inside=[Word(**{k:w.get(k) for k in ('word','start','end','score')}) for w in words
            if w.get('start') is not None and w.get('end') is not None and low<=w['start']<high]
    assign_speakers(inside,DiarizationResult(turns=[Turn(float(t['start']),float(t['end']),str(t['speaker'])) for t in turns]))
    changes=[];previous=None
    for word in sorted(inside,key=lambda w:w.start):
        if word.speaker is None:
            continue
        if previous is not None and word.speaker!=previous:
            changes.append(word.start)
        previous=word.speaker
    return changes

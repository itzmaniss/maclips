import pytest
from maclips.layouts import plan_layouts


def track(x,duration=10,shot=0):
    return {'median_box':[x,.2,.1,.2],'first_s':0,'last_s':duration,'shot_index':shot}


def plan(tracks):
    return plan_layouts({'1':{'shots':[{'shot_index':0,'start_s':0,'end_s':20}],'tracks':tracks}})['1']


def test_no_face_means_letterbox_only():
    assert list(plan([]))==['letterbox']


def test_one_face_centres_and_two_faces_split():
    one=plan([track(.2)])['face-centred']['segments'][0]
    assert one['x']==.25
    assert 'split' in plan([track(.1),track(.7)])


def test_three_faces_use_two_longest_duration_tracks():
    split=plan([track(.1,3),track(.4,10),track(.7,8)])['split']['segments'][0]
    assert [b[0] for b in split['boxes']]==[.4,.7]


def test_noncoexisting_track_fragments_do_not_make_split():
    t=track(.2,20);t['first_s']=12
    assert 'split' not in plan([track(.2,10),t])


def test_faces_too_close_for_separate_panels_use_longest_face_centred():
    stacked=[{'median_box':[.863,.812,.063,.112],'first_s':0,'last_s':10,'shot_index':0},
             {'median_box':[.859,.476,.055,.099],'first_s':0,'last_s':8,'shot_index':0}]
    plans=plan(stacked)
    assert 'split' not in plans
    assert plans['face-centred']['segments'][0]['x']==pytest.approx(.863+.063/2)
    assert 'split' not in plan([track(.4),track(.47)])

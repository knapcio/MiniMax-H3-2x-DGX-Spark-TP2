from h3_tp2 import cont_prompts as cp


def test_window_count_matches_render_sh():
    assert cp.window_count(5) == 1
    assert cp.window_count(10) == 1
    assert cp.window_count(30) == 3        # 736 frames -> 277 + 2 x 255
    assert cp.window_count(60) == 6
    assert cp.window_count(300) == 29
    assert cp.window_count(30, 345, 22) == 3
    assert cp.window_count(14.0, 345, 22) == 1   # 341 frames -> 345 (17k+5)


def test_prompts_shape_and_wording():
    ps = cp.continuation_prompts("A forest. The stream flows.", 3, "The light stays the same.")
    assert len(ps) == 3
    assert ps[0].startswith("A forest. The stream flows") and "One single continuous take" in ps[0]
    assert ps[0].endswith("The light stays the same.")
    assert ps[1] == ps[2] and ps[1].startswith(cp.REST_PREFIX) and "A forest. The stream flows." in ps[1]
    assert cp.continuation_prompts("x", 1) == ["x" + cp.FIRST_SUFFIX]


def test_request_duration():
    assert cp.request_duration({"duration": 30}) == 30.0
    assert cp.request_duration({"duration_seconds": 12.5}) == 12.5
    assert cp.request_duration({"target": {"duration_seconds": 8}}) == 8.0
    assert cp.request_duration({"duration": 30, "duration_seconds": 60, "target": {"duration_seconds": 300}}) == 300.0
    assert cp.request_duration({"duration": 30, "duration_seconds": 60, "target": {}}) == 60.0
    assert cp.request_duration({"duration": 30, "duration_seconds": 60, "target": {"duration_seconds": None}}) is None
    assert cp.request_duration({"duration": 30, "duration_seconds": None}) is None
    assert cp.request_duration({}) is None


def test_frame_count_when_duration_is_null_or_missing():
    assert cp.request_window_count({"duration": None}, 7200) == 29
    assert cp.request_window_count({"duration": 30, "target": {"duration_seconds": None}}, 7200) == 29
    assert cp.request_window_count({"duration": 30}, 7200) == 3  # duration overrides num_frames
    assert cp.request_window_count({}, 7200) == 29
    assert cp.request_window_count({}, 1) is None
    assert cp.request_window_count({}, None) is None
    assert cp.request_window_count({"continuation_window_frames": 175}, 7200) == 47


def test_rest_template_override():
    ps = cp.continuation_prompts("A forest.", 3, "", "Keep going. {base} Nothing new enters.")
    assert ps[1] == "Keep going. A forest. Nothing new enters." and ps[2] == ps[1]
    assert cp.continuation_prompts("A forest.", 2, "x", "Keep going.")[1] == "Keep going. x"

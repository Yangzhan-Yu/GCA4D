"""Verify the SAM3 subprocess bridge without needing a GPU or the model.

A stub interpreter stands in for $PYTHON_SAM3 and honours the same
``-m entrypoints.segment_with_sam3 --requests ... --output-json ...`` contract,
so this test covers request serialisation, environment wiring and result
parsing.  It does NOT cover SAM3 itself.

Run inside the gca environment::

    source scripts/gca_env.sh
    python tests/test_sam3_batch_plumbing.py
"""

import contextlib
import io
import json
import os
import stat
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from entrypoints.collect_vsibench_evidence import run_sam3_batch  # noqa: E402


STUB = '''#!/usr/bin/env python
"""Stub standing in for the SAM3 interpreter."""
import json, sys
from pathlib import Path

argv = sys.argv[1:]
assert argv[:2] == ['-m', 'entrypoints.segment_with_sam3'], argv
argv = argv[2:]
opts = {}
i = 0
while i < len(argv):
    key = argv[i].lstrip('-').replace('-', '_')
    opts[key] = argv[i + 1]
    i += 2
requests = json.loads(Path(opts['requests']).read_text())
prompts = requests['prompts']
frames = requests['frames']
results = {}
for index, frame in enumerate(frames):
    results[frame['frame_id']] = {
        prompt: [{
            'score': 0.9 - index * 0.1,
            'bbox': [1.0, 2.0, 30.0, 40.0],
            'mask_area_ratio': 0.05,
            'mask_path': f"/tmp/{frame['frame_id']}_{prompt}.png",
            'detector': 'sam3',
        }]
        for prompt in prompts
    }
payload = {
    'prompts': prompts,
    'model_loaded_seconds': 0.0,
    'inference_seconds': 0.0,
    'results': results,
    'errors': {},
}
Path(opts['output_json']).write_text(json.dumps(payload))
# echo the SAM3_ROOT the worker was given, so the test can assert on it
print('stub saw SAM3_ROOT=' + str(__import__('os').environ.get('SAM3_ROOT')))
'''


@dataclass
class FakeFrame:
    frame_id: str
    frame_path: str


def main():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        stub = root / 'fake_sam3_python'
        stub.write_text(STUB)
        stub.chmod(stub.stat().st_mode | stat.S_IEXEC)

        os.environ['SAM3_ROOT'] = '/opt/sam3-for-test'
        frames = [FakeFrame('001800', '/img/a.jpg'), FakeFrame('002160', '/img/b.jpg')]
        cache = run_sam3_batch(
            frames=frames,
            prompts=['chair', 'sofa'],
            work_dir=root / 'sam3',
            device='cuda',
            confidence=0.5,
            resolution=1008,
            max_per_prompt=1,
            python=str(stub),
        )

        assert set(cache) == {'001800', '002160'}, cache.keys()
        assert set(cache['001800']) == {'chair', 'sofa'}
        assert cache['001800']['chair'][0]['score'] == 0.9
        assert cache['002160']['chair'][0]['score'] == 0.8
        assert cache['001800']['sofa'][0]['detector'] == 'sam3'

        # the requests file the worker received must match our frames/prompts
        written = json.loads((root / 'sam3' / 'sam3_requests.json').read_text())
        assert written['prompts'] == ['chair', 'sofa']
        assert [f['frame_id'] for f in written['frames']] == ['001800', '002160']
        assert written['frames'][0]['image_path'] == '/img/a.jpg'

        # SAM3_ROOT must be forwarded so the worker can import the package.
        # run_sam3_batch echoes the worker's stdout, so capture it.
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            run_sam3_batch(
                frames=frames, prompts=['chair'], work_dir=root / 'envcheck',
                device='cuda', confidence=0.5, resolution=1008,
                max_per_prompt=1, python=str(stub),
            )
        assert 'stub saw SAM3_ROOT=/opt/sam3-for-test' in buffer.getvalue(), (
            buffer.getvalue()
        )

        # a failing worker must surface as a RuntimeError
        bad = root / 'bad_python'
        bad.write_text('#!/usr/bin/env python\nimport sys; sys.exit(3)\n')
        bad.chmod(bad.stat().st_mode | stat.S_IEXEC)
        try:
            run_sam3_batch(
                frames=frames, prompts=['chair'], work_dir=root / 'bad',
                device='cuda', confidence=0.5, resolution=1008,
                max_per_prompt=1, python=str(bad),
            )
        except RuntimeError as exc:
            assert 'returncode=3' in str(exc), exc
        else:
            raise AssertionError('failing worker should raise RuntimeError')

        # missing interpreter must fail with an actionable message
        saved = os.environ.pop('PYTHON_SAM3', None)
        try:
            run_sam3_batch(
                frames=frames, prompts=['chair'], work_dir=root / 'none',
                device='cuda', confidence=0.5, resolution=1008,
                max_per_prompt=1, python=None,
            )
        except RuntimeError as exc:
            assert 'PYTHON_SAM3' in str(exc), exc
        else:
            raise AssertionError('missing PYTHON_SAM3 should raise')
        finally:
            if saved is not None:
                os.environ['PYTHON_SAM3'] = saved

    print('PASS sam3 batch plumbing')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

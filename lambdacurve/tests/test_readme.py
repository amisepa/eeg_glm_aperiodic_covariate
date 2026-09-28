import re
from pathlib import Path

README = Path(__file__).resolve().parents[1] / "README.md"


def test_worked_example_runs():
    code = re.findall(r"```python\n(.*?)```", README.read_text(encoding="utf-8"), re.S)[0]
    ns = {}
    exec(compile(code, "README example", "exec"), ns)
    assert ns["verdict"](ns["r"]) == "reverses"
    assert ns["w"].lam_star > 1

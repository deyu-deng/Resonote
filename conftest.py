"""Keep pytest's scratch directories inside the repository.

Two reasons, both learned the hard way on the Windows dev machine: the default
temp root (C:\\Users\\...\\AppData\\Local\\Temp) held run dirs left behind by an
elevated pytest run that a normal run cannot even stat, so session teardown
raised WinError 5 *after* 214 tests had passed -- and a gate that dies while
reporting looks exactly like a green one. D:\\_POLICY also forbids scratch data
on C: at all.
"""

import pathlib

_ROOT = pathlib.Path(__file__).resolve().parent


def pytest_configure(config):
    if not config.option.basetemp:
        config.option.basetemp = str(_ROOT / ".pytest-tmp")
